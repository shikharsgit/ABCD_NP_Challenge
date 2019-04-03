import numpy as np
import os
import nibabel as nib
import nilearn 
from nilearn.image import load_img
import time
import matplotlib.pyplot as plt
import pandas as pd
import glob
from math import floor
from scipy.ndimage.interpolation import zoom
import torch
from torch.utils.data import Dataset,DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose


class ABCD_dataset(Dataset):
    """iq
    Dataset class for volumetric MRI image reconstruction of entire images
    """
    def __init__(self, data_df,mask,rn_list,
                 X_modality, y_modality="",
                 zoom_factor=None,
                 image_transform=None,
                 shape0 = [],
                 shape1 = [],
                 debug=False):

        self.data = data_df
        self.df = data_df
        self.mask = mask
        self.rn_list = rn_list
        self.min_edge = shape0
        self.max_edge = shape1
        self.X_modality = X_modality
        self.zoom_factor = zoom_factor
        self.image_transform = image_transform
        self.debug = debug

    @classmethod
    def fromcsv(cls,
                dataframe,
                mask,
                rn_list,
                data_dir,
                shape0 = [],
                shape1 = [],
                ID='SUBJECTKEY',
                X_modality='brain',
                zoom_factor=None,
                image_transform=None,
                debug=False):
        cls.data = cls._get_data_csv_df(dataframe, data_dir, ID)
        return cls(data_df=cls.data, mask=mask,rn_list=rn_list,shape0=shape0, shape1=shape1,X_modality=X_modality, zoom_factor=zoom_factor, image_transform=image_transform, debug=debug)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if idx >= len(self.data):
            raise IndexError   
        img_nii_X = nib.load(self.data[self.X_modality][idx])
        img_nii_X = img_nii_X.get_data().astype(np.float32)
        x0,y0,z0 = self.min_edge
        x1,y1,z1 = self.max_edge

        mask = self.mask
        rn_list = self.rn_list
        img_nii_X[np.isin(mask,rn_list,invert=True)]=0
        cropped_image = img_nii_X[x0:x1, y0:y1, z0:z1]
        #### Applying regional mask
        ### normalize the image to 0-1 values
        cropped_image = (cropped_image - np.min(cropped_image)) / (np.max(cropped_image) - np.min(cropped_image))
        # print(img_nii_X.shape)
        ID = self.data['subject'][idx]
        Y = self.data['target'][idx]
        X = cropped_image
        if self.zoom_factor is not None:
            X = zoom(X, self.zoom_factor)

        X = torch.from_numpy(X).unsqueeze(0).float()
        Y = torch.FloatTensor([Y])
        return {"image": X, "target": Y, "ID": ID}

    def get_subjectID(self, idx):
        return self.data.iloc[idx]

    @staticmethod
    def _get_data_csv_df(dataframe, data_dir, ID="SUBJECTKEY"):
        '''
        Creates a dataframe with following columns -
            "subject" : unique subject ID. Is also the index
                 "brain" : absolute path to the brain image
                 "gm_parc" : absolute path to the gm_parc image
        '''
        subs = []
        brain = []
        # gm_parc = []
        targets = []

        for subjectID in dataframe[ID]:
            brain_list = [glob.glob(os.path.join(data_dir, "submission_*"+str(subjectID) + "*_brain.nii.gz"))[0]]
            # gm_parc_list = list(glob.glob(os.path.join(data_dir, "submission_*"+str(subjectID)) + "*_gm_parc.nii.gz"))
            target_ID = dataframe[dataframe[ID]==subjectID]['residual_fluid_intelligence_score'].values[0]
            if brain_list:
                subs.extend([subjectID])
                brain.extend(brain_list)
                targets.extend([target_ID])
                # if t2_list:
                #     gm_parc.extend(gm_parc_list)
                # else:
                #     gm_parc.extend([[]])
            else:
                print("ID {} has no Brain Scan and will be excluded".format(subjectID))

        df = pd.DataFrame({"subject": subs,
                           "brain": brain,
                           "target" : targets})
        # print(df)
        return df





def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    '''https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py'''
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 3, 3),stride=stride,padding=(1,1,1),bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,1), stride=stride,bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    '''https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py'''
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dp1 = nn.Dropout3d(0.2)
        self.relu = nn.ReLU(inplace=True)  #could be ELU 
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dp2 = nn.Dropout3d(0.2)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dp1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dp2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class ResNet_abcd(nn.Module):
    '''https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py'''
    def __init__(self, block, layers, n_req_out,num_classes=1, zero_init_residual=False):
        super(ResNet_abcd, self).__init__()
        self.inplanes = 8
        self.conv1 = nn.Conv3d(1, 8, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(8,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dp1 = nn.Dropout3d(0.2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=5, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer1 = self._make_layer(block, 10, layers[0])
        self.layer2 = self._make_layer(block, 16, layers[1])
        self.layer3 = self._make_layer(block, 32, layers[2])
        self.layers = layers
        self.n_req_out = n_req_out
        # self.layer4 = self._make_layer(block, 64, layers[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1))
        # self.avgpool = nn.AvgPool3d(kernel_size=5, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        if layers[2]>0:
            n_linear = 32
        elif layers[1]>0:
            n_linear = 16
        else:
            n_linear = 10

        self.fc = nn.Linear(n_linear * n_req_out, num_classes)

        for m in self.modules(): ## Weight section
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode ='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion),
                nn.Dropout3d(0.2),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dp1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if self.layers[1]>0:
            x = self.layer2(x)
        
        if self.layers[2]>0:
            x = self.layer3(x)

        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
