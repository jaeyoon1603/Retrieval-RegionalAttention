import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
import copy
import Resnet
from tqdm import tqdm
import cv2
from Common import get_rmac_region_coordinates
from Common import pack_regions_for_network
from Common import L2Normalization
from Common import Shift
from Common import RoIPool
import warnings
warnings.simplefilter("ignore", DeprecationWarning)


class R_MACNet(nn.Module):
    def __init__(self):
        super(R_MACNet, self).__init__()

        self.l2norm = L2Normalization()
        self.r_mac_pool = RoIPool(1, 1, 0.03125)

        self._initialize_weights()

        self.resnet = Resnet.resnet101(pretrained=False)
        dic = torch.load('weights/resnet101_caffeProto.pth')
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

    def forward(self, x):
        batched_rois = [get_rmac_region_coordinates(x.shape[2], x.shape[3], 3) for i in range(x.shape[0])]
        region_size = len(batched_rois[0])
        rois = Variable(torch.FloatTensor(pack_regions_for_network(batched_rois)))

        h = x

        h = self.resnet(x)  # ( #batch, #channel, #h, #w)

        g = self.r_mac_pool(h, rois)
        g = g.squeeze(2).squeeze(2)  # (#batch * # regions, #channel)

        return g


class LandmarkDataset:
    def __init__(self, dir):
        self.dir = dir
        self.data_name_list = np.sort(os.listdir(self.dir + '/data'))
        self.cluster_id = map(lambda x: int(x.split('_')[0]), self.data_name_list)
        self.num_cluster = [0] * (self.cluster_id[-1] + 1)
        for i in range(len(self.cluster_id)):
            self.num_cluster[self.cluster_id[i]] += 1
        self.cluster = []
        for i in range(self.cluster_id[-1] + 1):
            offset = sum(self.num_cluster[:i])
            self.cluster.append(range(offset, offset + self.num_cluster[i]))


    def load_and_prepare_image(self, fname, S, means):
        # Read image, get aspect ratio, and resize such as the largest side equals S

        im = cv2.imread(fname)
        im_size_hw = np.array(im.shape[0:2])
        ratio = float(S) / np.max(im_size_hw)
        new_size = tuple(np.round(im_size_hw * ratio).astype(np.int32))
        im_resized = cv2.resize(im, (new_size[1], new_size[0]))


        I = im_resized.transpose(2, 0, 1) - means

        return I

    def __len__(self):
        return len(self.data_name_list)

    def __getitem__(self, idx):  


        means = np.array([103.93900299, 116.77899933, 123.68000031], dtype=np.float32)[None, :, None, None]

        img_name = os.path.join(self.dir + '/data', self.data_name_list[idx])
        image = self.load_and_prepare_image(img_name, 800, means)
        image = torch.from_numpy(image)

        return image

if __name__ == '__main__':
    gpu_num = 0
    
    dataSet_train = LandmarkDataset(dir='landmark/train')

    net = R_MACNet()
    net.eval()
    net = net.cuda(gpu_num)

    invertedList_train = []

    # Extract features of training dataset
    for i in tqdm(range(len(dataSet_train))):
        an_input = dataSet_train[i]
        #Filter out images that have too low resolution.
        if an_input.size(2) < 200 or an_input.size(3) < 200:
            continue
        variable_inputs = Variable(an_input.cuda(gpu_num), requires_grad=False, volatile=True)
        outputs = net(variable_inputs)
        for i in range(outputs.size(0)):
            feature = (outputs[i].cpu()).data.numpy()
            invertedList_train.append(np.array(feature))
    features_dataset = np.array(invertedList_train)
    
    #Calculate PCA parameters with only training datraset
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2048,whiten=True)
    pca.fit(features_dataset)
    import copy

    components = copy.deepcopy(pca.components_.T)
    components /= np.sqrt(pca.explained_variance_)
    components = components.T

    sh = Shift(2048)
    linear = nn.Linear(2048,2048,bias=False)
    linear.weight.data = torch.Tensor(components)
    sh.bias.data = torch.Tensor(-pca.mean_)

    np.save('weights/pca_components.npy', components)
    np.save('weights/pca_mean.npy', -pca.mean_)
