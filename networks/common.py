##### basic ####
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ConvTranspose2d
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class SumBl(nn.Module):
    def __init__(self, dimension=0):
        super(SumBl, self).__init__()
        self.d = dimension

    def forward(self, x):
        result = 0
        for i_item in x:
            result = result + i_item
        return result
    
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class TConv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(TConv, self).__init__()
        self.conv = nn.ConvTranspose2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU() 

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
    
class TConv2(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, output_padding = 1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(TConv2, self).__init__()
        self.conv = nn.ConvTranspose2d(c1, c2, k, s, autopad(k, p), groups=1, bias=False, output_padding = output_padding)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU() 

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
    

class ConvTranspose2d_2(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=1, output_padding = 1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvTranspose2d_2, self).__init__()
        self.conv = nn.ConvTranspose2d(c1, c2, k, s, autopad(k, p), groups=1, bias=False, output_padding = output_padding)

    def forward(self, x):
        return self.conv(x)

class ReShape(nn.Module):
    def __init__(self, img_shape = [3, 256, 256]):
        super(ReShape, self).__init__()
        self.img_shape = img_shape
    def forward(self, imgs):  
        return imgs.view(imgs.shape[0], *self.img_shape)

class SumConvTranspose2d(nn.Module):
    def __init__(self, in_ch, out_ch, kerner_size, stride, padding):
        super(SumConvTranspose2d, self).__init__()
        self.ConvTranspose = ConvTranspose2d(in_ch, out_ch, kerner_size, stride, padding)
    def forward(self, fea_list):  
        out_fea = 0
        for ifea in fea_list:
            out_fea = out_fea + ifea
        return self.ConvTranspose(out_fea)