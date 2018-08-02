import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import normalize
from os import listdir
from os.path import join
import Resnet
import random
from tqdm import tqdm
import cv2
from torchvision import transforms
from Common import get_rmac_region_coordinates
from Common import pack_regions_for_network
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
        self.sp_att_1 = nn.Softplus()
        self.conv_att_2 = nn.Conv1d(64, 1, 1, padding=0)
        self.sp_att_2 = nn.Softplus()
        

    def forward(self, features, rois):
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size()[0]
        
        outputs = Variable(torch.zeros(num_rois, num_channels*2,
                                       self.pooled_height,
                                       self.pooled_width))
        if features.is_cuda:
            outputs = outputs.cuda(torch.cuda.device_of(features).idx)
            
        # Based on roi pooling code of pytorch but, the only difference is to change max pooling to mean pooling
        for roi_ind, roi in enumerate(rois):
            batch_ind = int(roi[0].data[0])
            roi_start_w, roi_start_h, roi_end_w, roi_end_h = roi_start_w, roi_start_h, roi_end_w, roi_end_h = torch.round(roi[1:]* self.spatial_scale).data.cpu().numpy().astype(int)
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

                    is_empty = (hend <= hstart) or(wend <= wstart)
                    if is_empty:
                        outputs[roi_ind, :, ph, pw] = 0
                    else:
                        data = features[batch_ind]
                        # mean pooling with both of regional feature map and global feature map
                        outputs[roi_ind, :, ph, pw] = torch.cat((torch.mean(
                            torch.mean(data[:, hstart:hend, wstart:wend], 1, keepdim = True), 2, keepdim = True).view(-1)
                            ,torch.mean(
                            torch.mean(data, 1, keepdim = True), 2, keepdim = True).view(-1)), 0 )  # noqa
                        
        # Reshpae
        outputs = outputs.squeeze(2).squeeze(2)
        outputs = outputs.transpose(0,1).unsqueeze(0) # (1, # channel, #batch * # regions)
        #Calculate regional attention weights with context-aware regional feature vectors
        k = self.sp_att_1(self.conv_att_1(outputs))
        k = self.sp_att_2(self.conv_att_2(k)) # (1, 1, #batch * # regions)
        k = torch.squeeze(k,1)
        
        return k



class NetForTraining(nn.Module):


    def __init__(self):
        super(NetForTraining, self).__init__()
        
        #RoI max pooling
        self.r_mac_pool = RoIPool(1,1,0.03125)
        self.region_attention = ContextAwareRegionalAttentionNetwork(spatial_scale = 0.03125)
        self._initialize_weights()
        
        #This is for Imagenet classification. get the weights of fc from off-the-shelf Resnet101.
        self.fc = nn.Linear(2048,1000)
        dic = torch.load('models_pytorch/resnet101_caffeProto.pth', map_location=lambda storage, loc: storage)
        self.fc.weight.data = dic['fc.weight']
        self.fc.bias.data = dic['fc.bias']
        
        
        self.resnet = Resnet.resnet101(pretrained = False)
        dic= torch.load('models_pytorch/resnet101_caffeProto.pth', map_location=lambda storage, loc: storage)
        self.resnet.load_state_dict(dic, strict = False)
    
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
        batched_rois =  [get_rmac_region_coordinates(x.shape[2],x.shape[3],5) for i in range(x.shape[0])]
        region_size = len(batched_rois[0])
        rois = Variable(torch.FloatTensor(pack_regions_for_network(batched_rois)))
        
        h = x
         #Extract feature map
        h = self.resnet(x) #( #batch, #channel, #h, #w)
        

        
        g = self.r_mac_pool(h,rois) 
        g = g.squeeze(2).squeeze(2) # (#batch * # regions, #channel)
        
        g2 = self.region_attention(h,rois)
        g2 = g2.squeeze(0).squeeze(0)# (# batch * region)
        g = torch.mul(g.transpose(1,0),g2).transpose(1,0)  # regional weighted feature (# batch * region, #channel)
        
        g = g.contiguous()
        g = g.view(torch.Size([h.size(0), -1 ,h.size(1)])) # (#batch, # region, # channel)
        g = torch.transpose(g,1,2)    # (#batch * #channel, #region)
        

        g = torch.mean(g,2) #mean?

        g = self.fc(g)

        return g


class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self,dir,transform, means, train_or_val):
        super(ImageNetDataset, self).__init__()
        self.means = means
        self.transform = transform
        self.dir = dir
        if train_or_val == 'val':
            self.data_name_list = np.sort(os.listdir(dir+'/data'))
            self.data_name_list = np.sort(random.sample(self.data_name_list,10000))  # Reduce # of datas by sampling
            with open(dir+'/val.txt') as f:
                meta = f.readlines()
            meta = map(lambda x: x.split(), meta)
            self.cluster_id = []
            for i in range(len(self.data_name_list)):
                index = int(self.data_name_list[i].split('_')[2][:-5]) - 1 
                self.cluster_id = self.cluster_id + [int(meta[index][1])]
        elif  train_or_val == 'train':
            class_name_list =  np.sort(os.listdir(self.dir+'/data'))
            self.data_name_list = []
            self.cluster_id = []
            for i in range(len(class_name_list)):
                datas = map(lambda x: class_name_list[i]+'/'+x ,np.sort(os.listdir(self.dir+'/data/'+class_name_list[i])))
                self.data_name_list = self.data_name_list + datas# with class
                self.cluster_id = self.cluster_id + [i]*len(datas)
        
        

    
    def load_and_prepare_image(self, fname, S, means):
        # Read image, get aspect ratio, and resize such as the largest side equals S
    
        im = cv2.imread(fname)
        im_resized = np.array(self.transform(Image.fromarray(im)))
        I = im_resized.transpose(2, 0, 1) - means
        return I

    def __len__(self):
        return len(self.data_name_list)

    def __getitem__(self, idx): 
        img_name = os.path.join(self.dir+'/data', self.data_name_list[idx])
        image = self.load_and_prepare_image(img_name, 800, self.means)
        image = torch.from_numpy(image)
        return [image,torch.LongTensor([self.cluster_id[idx]])]

        
    
def __resize(img,size):
    #Set minimum dimension to size
    width = img.size[0]
    height = img.size[1]
    if width > height:
        new_height = size
        new_width  = int(new_height * width / height)
    else:
        new_width  = size
        new_height = int(new_width * height / width) 
    return img.resize((new_width,new_height))
    



        
def validation(net,dataSet_val ,gpu_num):

    
    data_transforms = transforms.Compose([
        transforms.Lambda(lambda x: __resize(x,800)),
        transforms.CenterCrop(800),

    ])
    

    dataSet = dataSet_val
    running_corrects = 0
    batchsize = 30
    for i in tqdm(range(int(len(dataSet)/batchsize))):
        inputs_list = []
        labels_list = []
        for j in range(int(batchsize)):
            data = dataSet[i*batchsize + j]   # randomly select
            inputs_list.append(data[0])
            labels_list.append(data[1])

        inputs = torch.cat(inputs_list,0)
        labels = torch.cat(labels_list,0)
        modelIn = Variable(inputs.cuda(gpu_num), requires_grad=False,volatile=True)
        labels = Variable(labels.cuda(gpu_num), requires_grad=False,volatile=True)

        outs = net(modelIn)
        _, preds = torch.max(outs.data, 1)
        running_corrects += torch.sum(preds == labels.data)
        
    return float(running_corrects)/float(len(dataSet))
        

if __name__ == '__main__':
    
    means = np.array([103.93900299, 116.77899933, 123.68000031], dtype=np.float32)[None, :, None, None]
    gpu_num = 0
    ImageNet_path = 'datasets/ImageNet/'
    transform_train = transforms.Compose([
        transforms.Lambda(lambda x: __resize(x,np.random.randint(10)+850)),
        transforms.RandomCrop(800),

    ])
    transform_val = transforms.Compose([
        transforms.Lambda(lambda x: __resize(x,800)),
        transforms.CenterCrop(800),

    ])
    
    dataSet_train = ImageNetDataset(dir= ImageNet_path + 'train',transform= transform_train, means = means, train_or_val = 'train')
    dataSet_val = ImageNetDataset(dir= ImageNet_path + 'val',transform= transform_val, means = means, train_or_val = 'val')
    miniBatch_size = 4
    
    net = NetForTraining()
    net.eval()
    net = net.cuda(gpu_num)

    init_lr = 0.001
    currentState = 'convNet'
    print('current state : ' + currentState)

    # Train only regional attention network
    optimizer_ft = optim.SGD(list(net.parameters())[:4], lr=init_lr, momentum=0.9, weight_decay = 0.00005)
    criterion =nn.CrossEntropyLoss()
    dataSet_size = len(dataSet_train)
    running_corrects = 0
    num_epoch = 1
    max_iterations = 999999
    optimizer_ft.zero_grad()
    for i in tqdm(range(max_iterations)):
        inputs_list = []
        labels_list = []
        input_num = []
        for j in range(int(miniBatch_size)):
            rand_num = np.random.randint(dataSet_size)
            input_num.append(rand_num)
            data = dataSet_train[rand_num]   # randomly select
            inputs_list.append(data[0])
            labels_list.append(data[1])

        inputs = torch.cat(inputs_list,0)
        labels = torch.cat(labels_list,0)
        modelIn = Variable(inputs.cuda(gpu_num))
        labels = Variable(labels.cuda(gpu_num))
        

        outs = net(modelIn)
        _, preds = torch.max(outs.data, 1)
        loss = torch.div(criterion(outs,labels),4)
        loss.backward()
        
        #Perform parameter-update with batch size of 16 
        if (i+1)% 4 ==0:
            optimizer_ft.step()
            optimizer_ft.zero_grad()

        running_corrects += torch.sum(preds == labels.data)


        #Make a check-point per 1600 iterations and validate the accuracy of network.
        if (i+1) % 1600 == 0:
            train_acc = float(running_corrects) / float(1600*miniBatch_size)
            val_acc = validation(net, dataSet_val, gpu_num)
            print('train Acc : ' + str(train_acc))
            print('val Acc : ' + str(val_acc))
            dic = net.region_attention.state_dict()
            torch.save(dic, 'weights/ContextAwareRegionalAttention_weights_'+str(num_epoch)+'_'+str(val_acc)+'.pth')
            running_corrects = 0
            num_epoch = num_epoch + 1
        
