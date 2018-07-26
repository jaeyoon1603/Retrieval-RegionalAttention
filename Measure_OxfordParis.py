import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
from os.path import join
import shutil
from sklearn.preprocessing import normalize
from collections import OrderedDict
import Resnet
from tqdm import tqdm
import cv2
import argparse
import subprocess
from Common import get_rmac_region_coordinates
from Common import pack_regions_for_network
from Common import L2Normalization
from Common import Shift
from Common import RoIPool
import warnings
warnings.simplefilter("ignore", DeprecationWarning)


class ContextAwareRegionalAttentionNetwork(nn.Module):
    def __init__(self, spatial_scale, pooled_height = 1, pooled_width = 1):
        super(ContextAwareRegionalAttentionNetwork, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

        self.conv_att_1 = nn.Conv1d(4096, 64, 1, padding=0)
        self.tanh = nn.Tanh()
        self.conv_att_2 = nn.Conv1d(64, 1, 1, padding=0)
        self.sp_att_2 = nn.Softplus()

    def forward(self, features, rois):
        
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size()[0]

        outputs = Variable(torch.zeros(num_rois, num_channels * 2,
                                       self.pooled_height,
                                       self.pooled_width))
        if features.is_cuda:
            outputs = outputs.cuda(torch.cuda.device_of(features).idx)

        # Based on roi pooling code of pytorch but, the only difference is to change max pooling to mean pooling
        for roi_ind, roi in enumerate(rois):
            batch_ind = int(roi[0].data[0])
            roi_start_w, roi_start_h, roi_end_w, roi_end_h = roi_start_w, roi_start_h, roi_end_w, roi_end_h = torch.round(
                roi[1:] * self.spatial_scale).data.cpu().numpy().astype(int)
            roi_width = max(roi_end_w - roi_start_w + 1, 1)
            roi_height = max(roi_end_h - roi_start_h + 1, 1)
            bin_size_w = float(roi_width) / float(self.pooled_width)
            bin_size_h = float(roi_height) / float(self.pooled_height)

            for ph in range(self.pooled_height):
                hstart = int(np.floor(ph * bin_size_h))
                hend = int(np.ceil((ph + 1) * bin_size_h))
                hstart = min(data_height, max(0, hstart + roi_start_h))
                hend = min(data_height, max(0, hend + roi_start_h))
                for pw in range(self.pooled_width):
                    wstart = int(np.floor(pw * bin_size_w))
                    wend = int(np.ceil((pw + 1) * bin_size_w))
                    wstart = min(data_width, max(0, wstart + roi_start_w))
                    wend = min(data_width, max(0, wend + roi_start_w))

                    is_empty = (hend <= hstart) or (wend <= wstart)
                    if is_empty:
                        outputs[roi_ind, :, ph, pw] = 0
                    else:
                        data = features[batch_ind]
                        # mean pooling with both of regional feature map and global feature map
                        outputs[roi_ind, :, ph, pw] = torch.cat((torch.mean(
                            torch.mean(data[:, hstart:hend, wstart:wend], 1, keepdim=True), 2, keepdim=True).view(-1)
                                                                 , torch.mean(
                            torch.mean(data, 1, keepdim=True), 2, keepdim=True).view(-1)), 0)  # noqa

        # Reshpae
        outputs = outputs.squeeze(2).squeeze(2)
        outputs = outputs.transpose(0, 1).unsqueeze(0)  # (1, # channel, #batch * # regions)
        #Calculate regional attention weights with context-aware regional feature vectors
        k = self.tanh(self.conv_att_1(outputs))
        k = self.sp_att_2(self.conv_att_2(k))  # (1, 1, #batch * # regions)
        k = torch.squeeze(k, 1)

        return k




class OurNet(nn.Module):
    def __init__(self):
        super(OurNet, self).__init__()

        self.l2norm = L2Normalization()
        #RoI max pooling
        self.r_mac_pool = RoIPool(1, 1, 0.03125)
        #Define regional attention network 
        self.region_attention = ContextAwareRegionalAttentionNetwork(spatial_scale = 0.03125)

        self.pca_shift = Shift(2048)
        self.pca_fc = nn.Linear(2048, 2048, bias=False)
        #Load the PCA weights learned with off-the-shelf Resnet101
        self.pca_fc.weight.data = torch.Tensor(np.load('weights/pca_components.npy'))
        self.pca_shift.bias.data = torch.Tensor(np.load('weights/pca_mean.npy'))

        self.resnet = Resnet.resnet101(pretrained=False)
        #Load off-the-shelf Resnet101 of caffe version.
        dic = torch.load('weights/resnet101_caffeProto.pth', map_location=lambda storage, loc: storage)
        self.resnet.load_state_dict(dic, strict=False)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, ):
        #Calculate R-MAC regions (Region sampling)
        batched_rois = [get_rmac_region_coordinates(x.shape[2], x.shape[3], 5) for i in range(x.shape[0])]
        region_size = len(batched_rois[0])
        rois = Variable(torch.FloatTensor(pack_regions_for_network(batched_rois)))

        h = x
        #Extract feature map
        h = self.resnet(x)  # ( #batch, #channel, #h, #w)

        #R-MAC module
        g = self.r_mac_pool(h, rois)
        g = g.squeeze(2).squeeze(2)  # (#batch * # regions, #channel)
        g = self.pca_shift(g)  # PCA
        g = self.pca_fc(g)
        g = self.l2norm(g)  # normalize each region
        
        #Regional attention module
        g2 = self.region_attention(h, rois)
        g2 = g2.squeeze(0).squeeze(0)  # (# batch * region)
        
        #Weighted mean
        g = torch.mul(g.transpose(1, 0), g2).transpose(1, 0)  # regional weighted feature (# batch * region, #channel)
        g = g.contiguous()
        g = g.view(torch.Size([h.size(0), -1, h.size(1)]))  # (#batch, # region, # channel)
        g = torch.transpose(g, 1, 2)  # (#batch * #channel, #region)
        g = torch.mean(g, 2)

        #Final L2
        g = self.l2norm(g)

        return g


class ImageHelper:
    def __init__(self, S, means):
        self.S = S
        self.means = means


    def get_features(self, I, net, gpu_num):

        output = net(Variable(torch.from_numpy(I).cuda(gpu_num)))
        output = np.squeeze(output.cpu().data.numpy())
        return output

    def load_and_prepare_image(self, fname, roi=None):
        # Read image, get aspect ratio, and resize such as the largest side equals S
        im = cv2.imread(fname)

        im_size_hw = np.array(im.shape[0:2])
        ratio = float(self.S) / np.max(im_size_hw)
        new_size = tuple(np.round(im_size_hw * ratio).astype(np.int32))
        im_resized = cv2.resize(im, (new_size[1], new_size[0]))
        # If there is a roi, adapt the roi to the new size and crop. Do not rescale
        # the image once again
        if roi is not None:
            roi = np.round(roi * ratio).astype(np.int32)
            im_resized = im_resized[roi[1]:roi[3], roi[0]:roi[2], :]
        # Transpose for network and subtract mean

        I = im_resized.transpose(2, 0, 1) - self.means
        return I


    
class Dataset:
    # This code is from Gordo et al at https://github.com/figitaki/deep-retrieval and slightly modified.
    def __init__(self, path, eval_binary_path):
        self.path = path
        self.eval_binary_path = eval_binary_path
        # Some images from the Paris dataset are corrupted. Standard practice is
        # to ignore them
        self.blacklisted = set(["paris_louvre_000136",
                                "paris_louvre_000146",
                                "paris_moulinrouge_000422",
                                "paris_museedorsay_001059",
                                "paris_notredame_000188",
                                "paris_pantheon_000284",
                                "paris_pantheon_000960",
                                "paris_pantheon_000974",
                                "paris_pompidou_000195",
                                "paris_pompidou_000196",
                                "paris_pompidou_000201",
                                "paris_pompidou_000467",
                                "paris_pompidou_000640",
                                "paris_sacrecoeur_000299",
                                "paris_sacrecoeur_000330",
                                "paris_sacrecoeur_000353",
                                "paris_triomphe_000662",
                                "paris_triomphe_000833",
                                "paris_triomphe_000863",
                                "paris_triomphe_000867"])
        self.load()

    def load(self):
        # Load the dataset GT
        self.lab_root = '{0}/lab/'.format(self.path)
        self.img_root = '{0}/jpg/'.format(self.path)
        lab_filenames = np.sort(os.listdir(self.lab_root))
        # Get the filenames without the extension
        self.img_filenames = [e[:-4] for e in np.sort(os.listdir(self.img_root)) if e[:-4] not in self.blacklisted]

        self.relevants = {}
        self.junk = {}
        self.non_relevants = {}

        self.filename_to_name = {}
        self.name_to_filename = OrderedDict()
        self.q_roi = {}
        for e in lab_filenames:
            if e.endswith('_query.txt'):
                q_name = e[:-len('_query.txt')]
                q_data = file("{0}/{1}".format(self.lab_root, e)).readline().split(" ")
                q_filename = q_data[0][5:] if q_data[0].startswith('oxc1_') else q_data[0]
                self.filename_to_name[q_filename] = q_name
                self.name_to_filename[q_name] = q_filename
                good = set([e.strip() for e in file("{0}/{1}_ok.txt".format(self.lab_root, q_name))])
                good = good.union(set([e.strip() for e in file("{0}/{1}_good.txt".format(self.lab_root, q_name))]))
                junk = set([e.strip() for e in file("{0}/{1}_junk.txt".format(self.lab_root, q_name))])
                good_plus_junk = good.union(junk)
                self.relevants[q_name] = [i for i in range(len(self.img_filenames)) if
                                          self.img_filenames[i] in good]
                self.junk[q_name] = [i for i in range(len(self.img_filenames)) if self.img_filenames[i] in junk]
                self.non_relevants[q_name] = [i for i in range(len(self.img_filenames)) if
                                              self.img_filenames[i] not in good_plus_junk]
                self.q_roi[q_name] = np.array(map(float, q_data[1:]), dtype=np.float32)

        self.q_names = self.name_to_filename.keys()
        self.q_index = np.array([self.img_filenames.index(self.name_to_filename[qn]) for qn in self.q_names])
        self.N_images = len(self.img_filenames)
        self.N_queries = len(self.q_index)

    def score(self, sim, temp_dir, eval_bin):
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        idx = np.argsort(sim, axis=1)[:, ::-1]
        maps = [self.score_rnk_partial(i, idx[i], temp_dir, eval_bin) for i in range(len(self.q_names))]
        for i in range(len(self.q_names)):
            print("{0}: {1:.2f}".format(self.q_names[i], 100 * maps[i]))
        print(20 * "-")
        print("Mean: {0:.1f}".format(round(100 * np.mean(maps), 1)))
        # print ("Mean: {0:.2f}".format(100 * np.mean(maps)))

    def score_rnk_partial(self, i, idx, temp_dir, eval_bin):
        rnk = np.array(self.img_filenames)[idx]
        with open("{0}/{1}.rnk".format(temp_dir, self.q_names[i]), 'w') as f:
            f.write("\n".join(rnk) + "\n")
        cmd = "{0} {1}{2} {3}/{4}.rnk".format(eval_bin, self.lab_root, self.q_names[i], temp_dir, self.q_names[i])
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        map_ = float(p.stdout.readlines()[0])
        p.wait()
        return map_

    def get_filename(self, i):
        return os.path.normpath("{0}/{1}.jpg".format(self.img_root, self.img_filenames[i]))

    def get_query_filename(self, i):
        return os.path.normpath("{0}/{1}.jpg".format(self.img_root, self.img_filenames[self.q_index[i]]))

    def get_query_roi(self, i):
        return self.q_roi[self.q_names[i]]

def extract_features(dataset, image_helper, net, gpu_num):

    dim_features = 2048
    N_queries = dataset.N_queries
    features_queries = np.zeros((N_queries, dim_features), dtype=np.float32)
    for i in tqdm(range(N_queries)):
        # Extract features for queries
        I = image_helper.load_and_prepare_image(dataset.get_query_filename(i),roi=dataset.get_query_roi(i))
        features_queries[i] = image_helper.get_features(I, net, gpu_num)

    dim_features = 2048
    N_dataset = dataset.N_images
    features_dataset = np.zeros((N_dataset, dim_features), dtype=np.float32)
    for i in tqdm(range(N_dataset)):
        # Extract features for dataset
        I = image_helper.load_and_prepare_image(dataset.get_filename(i), roi=None)
        features_dataset[i] = image_helper.get_features(I, net, gpu_num)
    return features_queries, features_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Oxford / Paris')
    parser.add_argument('--dataset', type=str, required=True, help='Choose the Oxford or Paris')
    parser.set_defaults(multires=False)
    args = parser.parse_args()
    
    gpu_num = 1
    S = 1024 #Maximum dimension
    dataset_path = 'datasets/' + args.dataset
    eval_binary = 'datasets/evaluation/compute_ap'
    temp_dir = 'temp'
    weight_path = 'weights/ContextAwareRegionalAttention_weights.pth'
    means = np.array([103.93900299, 116.77899933, 123.68000031], dtype=np.float32)[None, :, None, None]

    net = OurNet()
    net.eval()
    net = net.cuda(gpu_num)  
    
    #Load the trained weights of regional attention network
    dic = torch.load(weight_path)
    net.region_attention.load_state_dict(dic, strict=True)

    shutil.rmtree(temp_dir, ignore_errors=True)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    # Load the dataset and the image helper
    dataset = Dataset(dataset_path, eval_binary)
    image_helper = ImageHelper(S, means)


    # Extract features
    features_queries, features_dataset = extract_features(dataset, image_helper, net, gpu_num)

    # Compute similarity
    sim = features_queries.dot(features_dataset.T)
    dataset.score(sim, temp_dir, eval_binary)

    # query expansion with top-5 results
    top_k = 5
    idx = np.argsort(sim, axis=1)[:, ::-1]
    qe_features_queries = normalize(np.vstack(
        [np.vstack((features_queries[i], features_dataset[idx[i, :top_k]])).mean(axis=0) for i in
         range(len(features_queries))]))
    sim = qe_features_queries.dot(features_dataset.T)
    dataset.score(sim, temp_dir, eval_binary)

