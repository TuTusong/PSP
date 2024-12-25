import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from torch.autograd import Variable
import warnings
import numpy as np
import utils
from model import PANmodule
warnings.filterwarnings('ignore')




def tv_loss(x, beta = 0.5, reg_coeff = 5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    a,b,c,d=x.shape
    return reg_coeff*(torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))/(a*b*c*d))

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]



class weight_CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(weight_CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y,weight):
        diff = (x - y)*weight
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = (x - y)
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


# 计算特征提取模块的感知损失
def vgg16_loss(feature_module,loss_func,restored,target):
    res_fea=feature_module(restored).cuda( )
    tar_fea=feature_module(target).cuda( )
    loss=loss_func(res_fea,tar_fea).cuda( )
    return loss


# 获取指定的特征提取模块
def get_feature_module(layer_index,device=None):
    vgg = vgg16(pretrained=True, progress=True).features.cuda( )
    vgg.eval().cuda( )

    # 冻结参数
    for parm in vgg.parameters():
        parm.requires_grad = False

    feature_module = vgg[0:layer_index + 1].cuda( )
    feature_module.cuda(  )
    return feature_module


# 计算指定的组合模块的感知损失
class PerceptualLoss(nn.Module):
    def __init__(self,loss_func,layer_indexs=None):
        super(PerceptualLoss, self).__init__()
        self.creation=loss_func
        self.layer_indexs=layer_indexs

    def forward(self,restored,target):
        loss=0
        for index in self.layer_indexs:
            feature_module=get_feature_module(index).cuda( )
            loss+=vgg16_loss(feature_module,self.creation,restored,target).cuda( )
        return loss


CONTENT_LAYER = 'relu_16'
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG, self).__init__()
        self.features = make_layers(cfgs['E'], batch_norm=False) 
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.load_state_dict(torch.load('/home/ubuntu/szy/NAFNet_gray_plain/vgg19.pth'))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def vgg_19():
    vgg_19 = VGG().features
    model = nn.Sequential()

    i = 0
    for layer in vgg_19.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)
        if name == CONTENT_LAYER:
            break

    for param in model.parameters():
        param.requires_grad = False

    for param in vgg_19.parameters():
        param.requires_grad = False

    return model

def normalize_batch(batch):
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std

class VGGLoss(torch.nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.VGG_19 = vgg_19()
        self.L1_loss = torch.nn.L1Loss()

    def forward(self, img1, img2):
        img1 = F.interpolate(img1, scale_factor=0.5, mode="bilinear")
        img2 = F.interpolate(img2, scale_factor=0.5, mode="bilinear")
        img1_vgg = self.VGG_19(normalize_batch(img1))
        img2_vgg = self.VGG_19(normalize_batch(img2))
        loss_vgg = self.L1_loss(img1_vgg, img2_vgg)
        return loss_vgg