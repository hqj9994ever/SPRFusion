import torch
import torch.nn as nn

import math
import numbers

from collections import OrderedDict
from .cga import ChannelAttention
from .gmflow.gmflow import GMFlow
from model.mamba_simple import Mamba
from utils.utils import flow_warp2, to_3d, to_4d, weights_init



def define_A(in_channels, channels):
    netA = AttING(in_channels=in_channels, channels=channels)
    netA.apply(weights_init)
    return netA


def define_F():
    netF = align_FG()
    netF.apply(weights_init)
    return netF


def define_D(in_channles):
    netD = DualAttModule(in_dim=in_channles)
    netD.apply(weights_init)
    return netD


def define_G(in_channels):
    netG = FuseModule(in_channels=in_channels)
    netG.apply(weights_init)
    return netG


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape)==4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)
        

class PatchUnEmbed(nn.Module):
    def __init__(self,basefilter) -> None:
        super().__init__()
        self.nc = basefilter

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # BNC->BCHW
        return x
    

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, patch_size=4, stride=4, in_chans=36, embed_dim=32*32*32, norm_layer=None, flatten=True):
        super().__init__()
        
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = LayerNorm(embed_dim, 'BiasFree')

    def forward(self, x):
        #（b, c, h, w)->(b, c*s*p, h//s, w//s)
        # (b, h*w//s**2, c*s**2)
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x
    
    

class TokenSwapMamba(nn.Module):
    def __init__(self, dim):
        super(TokenSwapMamba, self).__init__()
        self.underencoder = Mamba(dim, bimamba_type=None)
        self.overencoder = Mamba(dim, bimamba_type=None)
        self.norm1 = LayerNorm(dim, 'with_bias')
        self.norm2 = LayerNorm(dim, 'with_bias')

    def forward(self, under, over, under_residual, over_residual):
        # under (B, N, C)
        # over (B, N, C)
        under_residual = under + under_residual
        over_residual = over + over_residual
        under = self.norm1(under_residual)
        over = self.norm2(over_residual)
        B, N, C = under.shape
        under_first_half = under[:, :, :C//2]
        over_first_half = over[:, :, :C//2]
        under_swap= torch.cat([over_first_half, under[:, :, C//2:]], dim=2)
        over_swap= torch.cat([under_first_half, over[:, :, C//2:]], dim=2)
        under_swap = self.underencoder(under_swap)
        over_swap = self.overencoder(over_swap)
        return under_swap, over_swap, under_residual, over_residual
    


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction):
        super(ChannelAttention, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        res = self.process(x)
        y = self.avg_pool(res)
        z = self.conv_du(y)
        return z *res + x



class Refine(nn.Module):

    def __init__(self, n_feat, out_channel):
        super(Refine, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_in = nn.Conv2d(n_feat*3, n_feat*2, 3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(n_feat*2, n_feat*2, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(n_feat*2, n_feat, 3, 1, 1)
        self.process = nn.Sequential(
            ChannelAttention(n_feat, 3))
        self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.lrelu(self.conv_in(x))
        out = self.lrelu(self.conv1(out))
        out = self.lrelu(self.conv2(out))
        out = self.lrelu(self.process(out))
        out = self.conv_last(out)

        return out


class AttING(nn.Module):
    def __init__(self, in_channels, channels):
        super(AttING, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2_1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.instance = nn.InstanceNorm2d(channels, affine=True)
        self.interative = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Sigmoid()
        )
        self.act = nn.LeakyReLU(0.1)

        self.process = nn.Sequential(nn.Conv2d(channels * 2, channels // 2, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels * 2, kernel_size=3, padding=1, bias=True),
                                     nn.Sigmoid())
        self.conv1x1 = nn.Conv2d(2 * channels, channels, 1, 1, 0)
        self.conv1x1_x1 = nn.Conv2d(channels, 3, 1, 1, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        out_instance = self.instance(x1)
        out_identity = x1
        out1 = self.conv2_1(out_instance)
        out2 = self.conv2_2(out_identity)

        out = torch.cat((out1, out2), 1)
        xp1 = self.interative(out) * out2 + out1
        xp2 = (1 - self.interative(out)) * out1 + out2
        xp = torch.cat((xp1, xp2), 1)
        xp = self.conv1x1(xp)
        xp = self.conv1x1_x1(xp)
        xout = xp

        return x1, xout
    

class align_FG(nn.Module):
    def __init__(self):
        super(align_FG, self).__init__()
        self.relu = nn.ReLU(True)

        self.flow_net = GMFlow(feature_channels=128,
                                num_scales=2,
                                upsample_factor=4,
                                num_head=1,
                                attention_type='swin',
                                ffn_dim_expansion=4,
                                num_transformer_layers=6,
                                )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x, ref, x_ins):
        ref_x = self.flow_net(ref, x, attn_splits_list=[2, 8],
                              corr_radius_list=[-1, 4],
                              prop_radius_list=[-1, 4],
                              )
        f = ref_x['flow_preds']
        F_warped_move_ = flow_warp2(x_ins, f[-1].permute(0, 2, 3, 1))

        x_ref = self.flow_net(x, ref, attn_splits_list=[2, 8],
                              corr_radius_list=[-1, 4],
                              prop_radius_list=[-1, 4],
                              )
        f_sup = x_ref['flow_preds']
        return F_warped_move_, f, f_sup
    


class DualAttModule(nn.Module):
    """ Interactive fusion module"""

    def __init__(self, in_dim=64):
        super(DualAttModule, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.key_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)

        self.gamma1 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.gamma2 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x, prior):
        x_q = self.query_conv(x)
        prior_k = self.key_conv(prior)
        energy = x_q * prior_k
        attention = self.sig(energy)
        attention_x = x * attention
        attention_p = prior * attention

        x_gamma = self.gamma1(torch.cat((x, attention_x), dim=1))
        x_out = x * x_gamma[:, [0], :, :] + attention_x * x_gamma[:, [1], :, :]

        p_gamma = self.gamma2(torch.cat((prior, attention_p), dim=1))
        prior_out = prior * p_gamma[:, [0], :, :] + attention_p * p_gamma[:, [1], :, :]

        return x_out, prior_out

    

class M3(nn.Module):
    def __init__(self, dim):
        super(M3, self).__init__()
        self.multi_modal_mamba_block = Mamba(dim, bimamba_type="m3")
        self.norm1 = LayerNorm(dim, 'with_bias')
        self.norm2 = LayerNorm(dim, 'with_bias')
        self.norm3 = LayerNorm(dim, 'with_bias')

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, I1, fusion_resi, I2, fusion, x_size):
        fusion_resi = fusion + fusion_resi
        fusion = self.norm1(fusion_resi)
        I2 = self.norm2(I2)
        I1 = self.norm3(I1)
        
        global_f = self.multi_modal_mamba_block(fusion, extra_emb1=I2, extra_emb2=I1)

        B, HW, C = global_f.shape
        fusion = global_f.transpose(1, 2).view(B, C, x_size[0], x_size[1])
        fusion =  (self.dwconv(fusion) + fusion).flatten(2).transpose(1, 2)
        return fusion, fusion_resi
    

class FuseModule(nn.Module):
    def __init__(self, in_channels):
        super(FuseModule, self).__init__()
        self.base_filter = in_channels
        self.stride = 1
        self.patch_size = 1
        self.embed_dim = self.base_filter * self.stride * self.patch_size

        self.patch_embed = PatchEmbed(in_chans=self.base_filter, embed_dim=self.embed_dim, patch_size=self.patch_size, stride=self.stride)
        self.patch_unembed = PatchUnEmbed(self.base_filter)
        self.channel_exchange1 = TokenSwapMamba(self.embed_dim)
        self.channel_exchange2 = TokenSwapMamba(self.embed_dim)

        self.shallow_fusion = nn.Conv2d(self.embed_dim*2, self.embed_dim, 3, 1, 1)

        self.M3_block1 = M3(self.embed_dim)
        self.M3_block2 = M3(self.embed_dim)
        self.M3_block3 = M3(self.embed_dim)

        self.refine = Refine(self.base_filter, 3)
    
    def forward(self, align_u, align_o):
        _, _, h, w = align_u.shape
        residual_under_f = 0
        residual_over_f = 0
        
        under_f = self.patch_embed(align_u)
        over_f = self.patch_embed(align_o)

        under_f, over_f, residual_under_f, residual_over_f = self.channel_exchange1(under_f, over_f, residual_under_f, residual_over_f)
        under_f, over_f, residual_under_f, residual_over_f = self.channel_exchange2(under_f, over_f, residual_under_f, residual_over_f)

        under_f = self.patch_unembed(under_f, (h, w))
        over_f = self.patch_unembed(over_f, (h, w))

        under_f = align_u
        over_f = align_o

        fusion_f = self.shallow_fusion(torch.cat([under_f, over_f], dim=1))

        under_f = self.patch_embed(under_f)
        over_f = self.patch_embed(over_f)
        fusion_f = self.patch_embed(fusion_f)

        residual_fusion_f = 0
        fusion_f, residual_fusion_f = self.M3_block1(under_f, residual_fusion_f, over_f, fusion_f, (h, w))
        fusion_f, residual_fusion_f = self.M3_block2(under_f, residual_fusion_f, over_f, fusion_f, (h, w))
        fusion_f, residual_fusion_f = self.M3_block3(under_f, residual_fusion_f, over_f, fusion_f, (h, w))

        fusion_f = self.patch_unembed(fusion_f, (h, w))

        output = self.refine(torch.cat([fusion_f, align_u, align_o], dim=1))

        return output
