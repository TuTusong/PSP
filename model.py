import torch
import torch.nn.functional as F
import torch.nn as nn
#from loss import MSELoss, CharbonnierLoss
import torch.nn.init as init
import functools
import numpy as np
import math
from torch.nn.modules.utils import _triple
import ipdb
import copy
from torch.nn.parameter import Parameter
import numbers
from einops import rearrange
from torch.nn import init
import matplotlib.pyplot as plt

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):  
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d): 
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def make_layer(block, n_layers):
    layers = [] 
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers) 

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))#对channnel进行平均

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5) 

# contrast-aware channel attention module
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=5):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class PA(nn.Module): 
    '''PA is pixel attention''' 
    def __init__(self, nf):
        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1) 
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)
        return out

class PAConv(nn.Module): 
    def __init__(self, nf, k_size=3):
        super(PAConv, self).__init__()
        self.k2 = nn.Conv2d(nf, nf, 1)  
        self.sigmoid = nn.Sigmoid()    
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  
    def forward(self, x):
        y = self.k2(x)
        y = self.sigmoid(y)
        out = torch.mul(self.k3(x), y) 
        out = self.k4(out)  
        return out

class SCPA(nn.Module):  
    def __init__(self, nf, reduction=2, stride=1, dilation=1): 
        super(SCPA, self).__init__()
        group_width = nf // reduction  
        self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)  
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False) 
        self.k1 = nn.Sequential(                                              
            nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                bias=False)                   
        )
        self.PAConv = PAConv(group_width)  
        self.conv3 = nn.Conv2d(
            group_width * reduction, nf, kernel_size=1, bias=False)  
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        residual = x
        out_a = self.conv1_a(x) 
        out_b = self.conv1_b(x) 
        out_a = self.lrelu(out_a) 
        out_b = self.lrelu(out_b)

        out_a = self.k1(out_a)
        out_b = self.PAConv(out_b)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b) 

        out = self.conv3(torch.cat([out_a, out_b], dim=1)) 
        out += residual

        return out
    
class SCPA_CCA(nn.Module):     
    def __init__(self, nf, reduction=2, stride=1, dilation=1): #reduction的作用讲通道数降为一半，方便后续进行connect
        super(SCPA_CCA, self).__init__()
        group_width = nf // reduction  #整数乘法，向下取整  （对应SC-PA第一步将通道数分半进行卷积）
        self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)  #nn.Conv2d进行卷积 
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False) 
        self.k1 = nn.Sequential(                                              ##nn.sequntial 进行容器的扩列
            nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                bias=False)                    #进行3×3的卷积 完成下面的部分
        )
        self.PAConv = PAConv(group_width)  #先1×1卷积，sigmod部分
        self.conv3 = nn.Conv2d(
            group_width * reduction, nf, kernel_size=1, bias=False)   #乘法的作用是（将两个向量进行connect）？？
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)#使用leakyRelu激活函数
        self.conv4= nn.Conv2d(nf,nf,1)
        self.cca = CCALayer(nf)#？
    def forward(self, x):
        residual = x
        out_a = self.conv1_a(x) #得到x'_{n-1}的部分
        out_b = self.conv1_b(x) #得到x''_{n-1}的部分
        out_a = self.lrelu(out_a) 
        out_b = self.lrelu(out_b)

        out_a = self.k1(out_a)#得到x''_{n}下面一行
        out_b = self.PAConv(out_b)#得到最上面一行
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b) 

        out = self.conv3(torch.cat([out_a, out_b], dim=1))  #将两个tensor拼接得到x_n上面一行
        out = self.conv4(self.cca(out))
        out += residual#完成逐元素进行相加

        return out


class half_blockPANmodule(nn.Module):  
    def __init__(self, in_nc=1, out_nc=1, nf=40, unf=32, nb=8):
        super(half_blockPANmodule, self).__init__()  
        SCPA_block_f = functools.partial(SCPA, nf=nf, reduction=2) 
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)  
        self.SCPA_trunk = make_layer(SCPA_block_f, nb)        
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv1 = nn.Conv2d(nf, unf, 3, 1, 1, bias=True)  
        self.att1 = PA(unf) 
        self.HRconv1 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True) 
        self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True) 
        self.att2 = PA(unf)
        self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
        self.conv_err = nn.Conv2d(unf, 1, 3, 1, 1, bias=True)  
        self.conv_last = nn.Conv2d(unf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)
    def calculate_err_gt(self, coarse_img, ground_truth, e=1e-6): 
        '''
        input:
        coarse_img:B C Hs Ws, tensor
        ground_truth:B C Hs Ws, tensor
        output:
        diff:B 1 H W, tensor
        '''

        diff = ((coarse_img - ground_truth) ** 2 + e) ** (1 / 2)
        diff = diff.sum(dim=1).unsqueeze(dim=1) 
        return diff
    
    def forward(self, x, gt=None):
        fea = self.conv_first(x) 
        trunk = self.trunk_conv(self.SCPA_trunk(fea)) 
        fea = fea + trunk 
        fea = self.upconv1(fea)
        fea = self.lrelu(self.att1(fea))
        fea = self.lrelu(self.HRconv1(fea))
        fea = self.upconv2(fea)
        fea = self.lrelu(self.att2(fea))
        out_feature = self.lrelu(self.HRconv2(fea))
        err = self.lrelu(self.conv_err(out_feature))
        out = self.conv_last(out_feature)
        base = x 
        out = out + base 
        if gt is not None:
            err_gt = self.calculate_err_gt(out, gt)  
            return out, base, out_feature, err_gt 
        else:
            return out, base, out_feature, err



class RED_CNN(nn.Module):
    def __init__(self, out_ch=96):
        super(RED_CNN, self).__init__()
        self.conv_first = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv_t1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv_t2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv_t3 = nn.ConvTranspose2d( out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv_t4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv_t_last = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        residual_1 = x
        out = self.relu(self.conv_first(x))
        out = self.relu(self.conv1(out))
        residual_2 = out
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        residual_3 = out
        out = self.relu(self.conv4(out))

        # decoder
        out = self.conv_t1(out)
        out += residual_3
        out = self.conv_t2(self.relu(out))
        out = self.conv_t3(self.relu(out))
        out += residual_2
        out = self.conv_t4(self.relu(out))
        out = self.conv_t_last(self.relu(out))
        out += residual_1
        out = self.relu(out)
        return out


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
    
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand  
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True) 
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True), 
        )

        # SimpleGate
        self.sg = SimpleGate() 

        ffn_channel = FFN_Expand * c 
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True) 

        self.norm1 = LayerNorm2d(c) 
        self.norm2 = LayerNorm2d(c) 

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity() 
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True) 
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True) 

    def forward(self, inp):
        x = inp
        
        x = self.conv1(x)
        x = self.conv2(x) 
        x = self.sg(x) 
        x = x * self.sca(x) 
        x = self.conv3(x) 

        x = self.dropout1(x)）

        y = self.norm1(inp + x * self.beta) #post norm

        x = self.conv4(y) 
        x = self.sg(x) 
        x = self.conv5(x) 

        x = self.dropout2(x)
        z = self.norm2(y + x * self.gamma) 

        return z


class NAFNet(nn.Module):

    def __init__(self, img_channel=1, width=32, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True) 
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True) 

        self.encoders = nn.ModuleList() 
        self.decoders = nn.ModuleList() 
        self.middle_blks = nn.ModuleList() 
        self.ups = nn.ModuleList() 
        self.downs = nn.ModuleList() 

        chan = width 
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential( 
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2) 
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)] 
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),  
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2 
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)] 
                )
            )

        self.padder_size = 2 ** len(self.encoders) 
        y = 2**len(self.encoders)
        print(y)
        print(len(self.encoders))

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp) 

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size() 
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size 
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h)) #对图片进行填充
        return x





class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x): 
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
        

class eMSM_T(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(eMSM_T, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.position_embedding=PositionalEncoding(d_model=dim)
        
        self.project_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(0.)
        )

    def forward(self, x):
        b,c,t,h,w=x.shape

        x=F.adaptive_avg_pool3d(x,(t,1,1))

        x=x.squeeze(-1).squeeze(-1).permute(2,0,1) #t,b,c

        x= self.position_embedding(x).permute(1,0,2) #b,t,c

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        #ipdb.set_trace()

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.num_heads), (q, k, v))

        scale = (c//self.num_heads) ** -0.5
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * scale

        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.num_heads)

        out = self.project_out(out).permute(0,2,1)

        return out


class eMSM_I(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(eMSM_I, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        b,c,t,h,w = x.shape
        x=F.adaptive_avg_pool3d(x,(1,h,w))
        x=x.permute(0,1,3,4,2).squeeze(-1)

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

         
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    

class LITFormerBlock(nn.Module):
    def __init__(self,input_channel,output_channel,num_heads_s=8,num_heads_t=2,kernel_size=1,stride=1,padding=0,
                groups=1,bias=False,res=True,attention_s=False,attention_t=False):
        super().__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        assert len(kernel_size) == len(stride) == len(padding) == 3
        self.input_channel=input_channel
        self.output_channel=output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.res=res
        self.attn_s=attention_s
        self.attn_t=attention_t
        self.num_heads_s=num_heads_s
        self.num_heads_t=num_heads_t
        self.activation=nn.LeakyReLU(inplace=True)

        if attention_s==True:
            self.attention_s=eMSM_I(dim=input_channel, num_heads=num_heads_s, bias=False)
        self.conv_1x3x3=nn.Conv3d(input_channel,output_channel,kernel_size=(1, kernel_size[1], kernel_size[2]),
                            stride=(1, stride[1], stride[2]),padding=(0, padding[1], padding[2]),groups=groups)
        if attention_t==True:
            self.attention_t=eMSM_T(dim=input_channel, num_heads=num_heads_t, bias=False)
        self.conv_3x1x1=nn.Conv3d(input_channel,output_channel,kernel_size=(kernel_size[0], 1, 1),
                            stride=(stride[0], 1, 1),padding=(padding[0], 0, 0),groups=groups)

        if self.input_channel != self.output_channel:
            self.shortcut=nn.Conv3d(in_channels=input_channel,out_channels=output_channel,kernel_size=1,padding=0,stride=1,groups=1,bias=False)


    def forward(self, inputs):

        if self.attn_s==True or self.attn_t==True:

            attn_s=self.attention_s(inputs).unsqueeze(2)  if self.attn_s==True else 0 
            attn_t=self.attention_t(inputs).unsqueeze(-1).unsqueeze(-1) if self.attn_t==True else 0

            inputs_attn=inputs+attn_t+attn_s

            conv_S=self.conv_1x3x3(inputs_attn)
            conv_T=self.conv_3x1x1(inputs_attn)

            if self.input_channel == self.output_channel: 
                identity_out=inputs_attn 
            else: 
                identity_out=self.shortcut(inputs_attn)

        else:
            if self.input_channel == self.output_channel: 
                identity_out=inputs 
            else: 
                identity_out=self.shortcut(inputs)
                
            conv_S=self.conv_1x3x3(inputs)
            conv_T=self.conv_3x1x1(inputs)

        if self.res:
            output=conv_S+conv_T+identity_out
        elif not self.res:
            output=conv_S+conv_T  

        return output


class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels,num_heads_s=8,num_heads_t=2,
                res=True,attention_s=False,attention_t=False):
        super(DoubleConv,self).__init__()
        self.double_conv=nn.Sequential(
            LITFormerBlock(in_channels,in_channels,num_heads_s=num_heads_s,num_heads_t=num_heads_t,res=res,
                            attention_s=attention_s,attention_t=attention_t),
            nn.LeakyReLU(inplace=True),
            LITFormerBlock(in_channels,out_channels,res=res),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self,x):
        return self.double_conv(x)
      
               
class Down(nn.Module):

    def __init__(self, in_channels, out_channels,num_heads_s=8,num_heads_t=2,
                 res=True,attention_s=False,attention_t=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d((1,2,2), (1,2,2)),
            DoubleConv(in_channels, out_channels,num_heads_s=num_heads_s,num_heads_t=num_heads_t,
                       res=res,attention_s=attention_s,attention_t=attention_t)
        )
            
    def forward(self, x):
        return self.encoder(x)

    
class LastDown(nn.Module):

    def __init__(self, in_channels, out_channels,num_heads_s=8,num_heads_t=2,
                res=True,attention_s=False,attention_t=False):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.MaxPool3d((1,2,2), (1,2,2)),
            LITFormerBlock(in_channels,2*in_channels,num_heads_s=num_heads_s,num_heads_t=num_heads_t,res=res,
                      attention_s=attention_s,attention_t=attention_t),
            nn.LeakyReLU(inplace=True),
            LITFormerBlock(2*in_channels,out_channels),
            nn.LeakyReLU(inplace=True),
            )
    def forward(self, x):
        return self.encoder(x)


    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels,res_unet=True,trilinear=True, num_heads_s=8,num_heads_t=2,
                res=True,attention_s=False,attention_t=False):
        super().__init__()
        self.res_unet=res_unet
        if trilinear:
            self.up = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels , kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_channels, out_channels,num_heads_s=num_heads_s,num_heads_t=num_heads_t,
                               res=res,attention_s=attention_s,attention_t=attention_t)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.res_unet:
            x=x1+x2
        else:
            x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels,res=True,activation=False):
        super().__init__()
        self.act=activation
        self.conv =LITFormerBlock(in_channels, out_channels,res=res)
        self.activation = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        x=self.conv(x)
        if self.act==True:
            x=self.activation(x)
        return x
        
        
class LITFormer(nn.Module):
    def __init__(self, in_channels=1,out_channels=1,n_channels=64,num_heads_s=[1,2,4,8],num_heads_t=[1,2,4,8],
                res=True,attention_s=False,attention_t=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_channels = n_channels
        
        self.firstconv=SingleConv(in_channels, n_channels//2,res=res,activation=True)
        self.enc1 = DoubleConv(n_channels//2, n_channels,num_heads_s=num_heads_s[0],num_heads_t=num_heads_t[0],
                               res=res,attention_s=attention_s,attention_t=attention_t) 
        
        self.enc2 = Down(n_channels, 2 * n_channels,num_heads_s=num_heads_s[1],num_heads_t=num_heads_t[1],
                               res=res,attention_s=attention_s,attention_t=attention_t)
        
        self.enc3 = Down(2 * n_channels, 4 * n_channels,num_heads_s=num_heads_s[2],num_heads_t=num_heads_t[2],
                               res=res,attention_s=attention_s,attention_t=attention_t)
        
        self.enc4 = LastDown(4 * n_channels, 4 * n_channels,num_heads_s=num_heads_s[3],num_heads_t=num_heads_t[3],
                             res=res,attention_s=attention_s,attention_t=attention_t)
        
        self.dec1 = Up(4 * n_channels, 2 * n_channels,num_heads_s=num_heads_s[2],num_heads_t=num_heads_t[2],
                               res=res,attention_s=attention_s,attention_t=attention_t)
        
        self.dec2 = Up(2 * n_channels, 1 * n_channels,num_heads_s=num_heads_s[1],num_heads_t=num_heads_t[1],
                               res=res,attention_s=attention_s,attention_t=attention_t)
        
        self.dec3 = Up(1 * n_channels, n_channels//2,num_heads_s=num_heads_s[0],num_heads_t=num_heads_t[0],
                               res=res,attention_s=attention_s,attention_t=attention_t)
        self.out1 = SingleConv(n_channels//2,n_channels//2,res=res,activation=True)
        self.depth_up = nn.Upsample(scale_factor=tuple([1,1,1]),mode='trilinear')
        self.out2 = SingleConv(n_channels//2,out_channels,res=res,activation=False)

    def forward(self, x):
        x = x.unsqueeze(2)
        b,c,d,h,w=x.shape
        x =self.firstconv(x)
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        output = self.dec1(x4, x3)
        output = self.dec2(output, x2)
        output = self.dec3(output, x1)
        output = self.out1(output)+x
        output = self.depth_up(output)
        output = self.out2(output)
        output = output.squeeze(2)
        return output
    
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channel):
        super(ConvBlock, self).__init__()
        self.body = nn.Sequential(
            nn.BatchNorm2d(in_channel,eps=1.001e-5),
            nn.ReLU(),
            nn.Conv2d(in_channel,80,kernel_size=(1,1),padding=(0,0) ,bias=False),
            nn.BatchNorm2d(80,eps=1.001e-5),
            nn.ReLU(),
            nn.Conv2d(80,16,kernel_size=(3,3),padding=(1,1), bias=False),
        )
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.normal_(m.weight,mean=0,std=0.01)

    def forward(self, x):
        x1 = self.body(x)
        out = torch.cat([x,x1], dim=1)

        return out


class TransitionBlock(torch.nn.Module):
    def __init__(self):
        super(TransitionBlock, self).__init__()
        self.body = nn.Sequential(
            nn.BatchNorm2d(80,eps=1.001e-5),
            nn.ReLU(),
            nn.Conv2d(80,16, kernel_size=(1,1))
        )
    def forward(self, x):
        x = self.body(x)
        return x