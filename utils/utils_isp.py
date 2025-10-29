# coding=utf-8

import numpy as np
from scipy.interpolate import interp1d
import torch
import torch.nn as nn

from utils import utils_color
from utils import utils_noise
from utils import utils_blur
from utils import utils_image as util


class Demosaic(nn.Module):
    """ matlab demosaicking
    Args:
        x: Nx1xWxH with RGGB bayer pattern

    Returns:
        output: Nx3xWxH
    """
    def __init__(self, NeedDemosaic=True):
        super(Demosaic, self).__init__()
        self.NeedDemosaic = NeedDemosaic
        if self.NeedDemosaic:
            kgrb = 1/8*torch.FloatTensor([[0, 0, -1, 0, 0],
                                          [0, 0, 2, 0, 0],
                                          [-1, 2, 4, 2, -1],
                                          [0, 0, 2, 0, 0],
                                          [0, 0, -1, 0, 0]])    
            krbg0 = 1/8*torch.FloatTensor([[0, 0, 1/2, 0, 0],
                                           [0, -1, 0, -1, 0],
                                           [-1, 4, 5, 4, -1],
                                           [0, -1, 0, -1, 0],
                                           [0, 0, 1/2, 0, 0]])
            krbg1 = krbg0.t()
            krbbr = 1/8*torch.FloatTensor([[0, 0, -3/2, 0, 0],
                                           [0, 2, 0, 2, 0],
                                           [-3/2, 0, 6, 0, -3/2],
                                           [0, 2, 0, 2, 0],
                                           [0, 0, -3/2, 0, 0]])
            self.register_buffer('k', torch.stack((kgrb, krbg0, krbg1, krbbr), 0).unsqueeze(1))

    def forward(self, x):
        if self.NeedDemosaic:
            output = x.repeat(1, 3, 1, 1)
            x = nn.functional.pad(x, (2, 2, 2, 2), mode='reflect')
            conv_cfa = nn.functional.conv2d(x, self.k, padding=0, bias=None)
            output[:, 1, 0::2, 0::2] = conv_cfa[:, 0, 0::2, 0::2]
            output[:, 1, 1::2, 1::2] = conv_cfa[:, 0, 1::2, 1::2]
            output[:, 0, 0::2, 1::2] = conv_cfa[:, 1, 0::2, 1::2]
            output[:, 0, 1::2, 0::2] = conv_cfa[:, 2, 1::2, 0::2]
            output[:, 0, 1::2, 1::2] = conv_cfa[:, 3, 1::2, 1::2]
            output[:, 2, 0::2, 1::2] = conv_cfa[:, 2, 0::2, 1::2]
            output[:, 2, 1::2, 0::2] = conv_cfa[:, 1, 1::2, 0::2]
            output[:, 2, 0::2, 0::2] = conv_cfa[:, 3, 0::2, 0::2]
        else:
            output = torch.cat((x[:, 0:1, 0::2, 0::2], (x[:, 0:1, 0::2, 1::2] + x[:, 0:1, 1::2, 0::2]) / 2, x[:, 0:1, 1::2, 1::2]), 1)
        return output.clamp_(0, 1)
    
    def reverse(self, x):
        if self.NeedDemosaic:
            output = torch.zeros_like(x[:, 1:2, :, :])
            output[:, 0, 0::2, 0::2] = x[:, 0, 0::2, 0::2]
            output[:, 0, 0::2, 1::2] = x[:, 1, 0::2, 1::2]
            output[:, 0, 1::2, 0::2] = x[:, 1, 1::2, 0::2]
            output[:, 0, 1::2, 1::2] = x[:, 2, 1::2, 1::2]
        else:
            output = torch.zeros(x.size(0), x.size(1)//3, x.size(2)*2, x.size(3)*2).type_as(x)
            output[:, 0, 0::2, 0::2] = x[:, 0, ...]
            output[:, 0, 0::2, 1::2] = x[:, 1, ...]
            output[:, 0, 1::2, 0::2] = x[:, 1, ...]
            output[:, 0, 1::2, 1::2] = x[:, 2, ...]
        return output.clamp_(0, 1)


class AddNoise(nn.Module):
    '''
    Add shot and read noise, Nx1xWxH -> Nx1xWxH
    '''
    def __init__(self, p_noise=0.75):
        super(AddNoise, self).__init__()
        self.p_noise = p_noise
        self.add_or_not = torch.rand(1) < self.p_noise

    def forward(self, x):
        return x
    
    def reverse(self, x):
        if self.add_or_not:
            noise = torch.zeros_like(x)
            x, noise, noise_level = utils_noise.add_rawnoise(x)
        else:
            noise_level = torch.Tensor([0.0])
        return x, noise, noise_level
    

class AddBlur(nn.Module):
    """
    Add blur, Nx3xWxH -> Nx3xWxH
    """
    def __init__(self, p_blur=0.5):
        super(AddBlur, self).__init__()
        self.p_blur = p_blur
        self.add_or_not = torch.rand(1) < self.p_blur

        kernel = utils_blur.blur_kernel()
        kernel = util.single2tensor4(kernel[..., None])
        self.register_buffer('k', kernel)

    def forward(self, x):
        return x
    
    def reverse(self, x):
        if self.add_or_not:
            n, c = x.shape[:2]
            p1, p2 = (self.k.shape[-2]-1)//2, (self.k.shape[-1]-1)//2
            x = torch.nn.functional.pad(x, pad=(p1, p2, p1, p2), mode='replicate')
            self.k = self.k.repeat(n, c, 1, 1)
            self.k = self.k.view(-1, 1, self.k.shape[2], self.k.shape[3])
            x = x.view(1, -1, x.shape[2], x.shape[3])
            x = torch.nn.functional.conv2d(x, self.k, bias=None, stride=1, padding=0, groups=n*c)
            x = x.view(n, c, x.shape[2], x.shape[3])
        return x.clamp_(0, 1)
    


class ExposureCompensationWhiteBalance(nn.Module):
    '''
    Exposure Compensation and white balance 
    Exposure: BaselineExposure: from camera profile
    '''    
    def __init__(self, Exposure=0, R_gain=torch.FloatTensor([2.0]), G_gain=torch.FloatTensor([1.0]), B_gain=torch.FloatTensor([1.8]), Inflection=0.9):
        super(ExposureCompensationWhiteBalance, self).__init__()
        self.r_gain, self.g_gain, self.b_gain = R_gain, G_gain, B_gain
        self.exposure = Exposure
        self.inflection = Inflection
        self.register_buffer('gains', torch.tensor([self.r_gain, self.g_gain, self.b_gain]).mul(2**self.exposure).view(1, 3, 1, 1))
        self.safe_gains = None

    def forward(self, x):
        return x.mul_(self.gains).clamp_(0, 1) if self.safe_gains is None else x.div_(self.safe_gains).clamp_(0, 1)
    
    def reverse(self, x):
        gray = torch.mean(x, dim=1, keepdim=True)
        mask = (torch.clamp(gray - self.inflection, min=0.0) / (1.0 - self.inflection)) ** 2.0
        self.safe_gains = torch.maximum(mask + (1.0 - mask) / self.gains, 1 / self.gains)
        return x.mul_(self.safe_gains).clamp_(0, 1)
    


class Raw2XYZ(nn.Module):
    """
    camera raw (after demosaicing+exposure&white gain) --> XYZ(D50)
    """  
    def __init__(self, weight):
        super(Raw2XYZ, self).__init__()
        weight_inv = torch.inverse(weight).unsqueeze(-1).unsqueeze(-1)
        self.register_buffer('weight', weight.unsqueeze(-1).unsqueeze(-1))
        self.register_buffer('weight_inv', weight_inv)

    def forward(self, x):
        return torch.matmul(x.permute(0, 2, 3, 1), self.weight.squeeze().t()).permute(0, 3, 1, 2)

    def reverse(self, x):
        return torch.matmul(x.permute(0, 2, 3, 1), self.weight_inv.squeeze().t()).permute(0, 3, 1, 2)



class XYZ2LinearRGB(nn.Module):
    """
    XYZ(D50) --> linear sRGB(D65)
    """
    def __init__(self):
        super(XYZ2LinearRGB, self).__init__()
        weight = utils_color.xyz2linearrgb_weight(0, is_forward=True)
        weight_inv = torch.inverse(weight).unsqueeze(-1).unsqueeze(-1)
        self.register_buffer('weight', weight.unsqueeze(-1).unsqueeze(-1))
        self.register_buffer('weight_inv', weight_inv)

    def forward(self, x):
        return nn.functional.conv2d(x, self.weight, bias=None)

    def reverse(self, x):
        return nn.functional.conv2d(x, self.weight_inv, bias=None)



class ToneMapping(nn.Module):
    '''
    Tone Mapping
    '''
    def __init__(self, ToneCurveX, ToneCurveY, delta=1e-6):
        super(ToneMapping, self).__init__()
        self.delta = delta
        xi = np.linspace(0, 1, num=int(1/delta+1), endpoint=True)
        yi = interp1d(ToneCurveX, ToneCurveY, kind='cubic')(xi)
        yi_inv = interp1d(yi, xi, kind='cubic')(xi)          
        self.register_buffer('yi', torch.from_numpy(yi).float())
        self.register_buffer('yi_inv', torch.from_numpy(yi_inv).float())

    def forward(self, x):
        x = self.yi[(torch.round(x.clamp_(0, 1) / self.delta)).long()]
        return x.clamp_(0, 1)
    
    def reverse(self, x):
        x = self.yi_inv[(torch.round(x.clamp_(0, 1) / self.delta)).long()]
        return x.clamp_(0, 1)



class GammaCorrect(nn.Module):
    """
    Gamma correction
    linear RGB --> sRGB
    """
    def __init__(self):
        super(GammaCorrect, self).__init__()

    def forward(self, x):
        idx = x > 0.0031308
        x[idx] = 1.055*torch.pow(x[idx], 1./2.4)-0.055
        x[~idx] = 12.92*x[~idx]
        return x.clamp_(0, 1)

    def reverse(self, x):
        idx = x > 0.04045
        x[~idx] = x[~idx]/12.92
        x[idx] = torch.pow((200.0*x[idx] + 11.0)/211.0, 2.4)
        return x.clamp_(0, 1)



class ISP(nn.Module):
    def __init__(self, weight_raw2xyz, ToneCurveX, ToneCurveY, NeedDemosaic=True, Exposure=0, R_gain=2.0, G_gain=1.0, B_gain=1.8, Inflection=0.9, p_noise=0.75, p_blur=0.5):
        super(ISP, self).__init__()
        self.demosaic = Demosaic(NeedDemosaic=NeedDemosaic)
        self.addnoise = AddNoise(p_noise=p_noise)
        self.addblur = AddBlur(p_blur=p_blur)
        self.exposurecompensationwhitebalance = ExposureCompensationWhiteBalance(Exposure=Exposure, R_gain=R_gain, G_gain=G_gain, B_gain=B_gain, Inflection=Inflection)
        self.raw2xyz = Raw2XYZ(weight=weight_raw2xyz)
        self.xyz2linearrgb = XYZ2LinearRGB()
        self.tonemapping = ToneMapping(ToneCurveX=ToneCurveX, ToneCurveY=ToneCurveY)
        self.gammacorrect = GammaCorrect()

    def forward(self, x):
        # x = self.demosaic.forward(x)
        x = self.exposurecompensationwhitebalance.forward(x)
        x = self.raw2xyz.forward(x)
        x = self.xyz2linearrgb.forward(x)
        x = self.tonemapping.forward(x)
        x = self.gammacorrect.forward(x)
        return x
    
    def reverse(self, x):
        x = self.gammacorrect.reverse(x)
        x = self.tonemapping.reverse(x)
        x = self.xyz2linearrgb.reverse(x)
        x = self.raw2xyz.reverse(x)
        x = self.exposurecompensationwhitebalance.reverse(x)
        # x = self.addblur.reverse(x)
        # x = self.demosaic.reverse(x)
        # x = self.addnoise.reverse(x)
        return x
            