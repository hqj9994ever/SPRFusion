import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .backbone import CNNEncoder
from .transformer import FeatureTransformer, FeatureFlowAttention
from .matching import global_correlation_softmax, local_correlation_softmax
from .geometry import flow_warp
from .utils import normalize_img, feature_add_position
from model.SAM_encoder import get_encoder_base
from utils import utils_image as util
from .visualize_tool import visualize_mean_map


class ResidualBlockNoBN(nn.Module):
    def __init__(self, num_feat=64, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class make_dilation_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dilation_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2 + 1,
                              bias=True, dilation=2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class DRDB(nn.Module):
    def __init__(self, nChannels, denseLayer, growthRate):
        super(DRDB, self).__init__()
        num_channels = nChannels
        modules = []
        for i in range(denseLayer):
            modules.append(make_dilation_dense(num_channels, growthRate))
            num_channels += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(num_channels, nChannels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1 or in_planes != planes:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1 or in_planes != planes:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1 or in_planes != planes:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1 or in_planes != planes:
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes), 
                nn.Conv2d(planes, planes, kernel_size=1, stride=stride),
                self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)
        


class GMFlow(nn.Module):
    def __init__(self,
                 num_scales=1,
                 upsample_factor=8,
                 feature_channels=128,
                 attention_type='swin',
                 num_transformer_layers=6,
                 ffn_dim_expansion=4,
                 num_head=1,
                 sam_checkpoint='model_zoo/ckpt/sam_vit_b_01ec64.pth',
                 **kwargs,
                 ):
        super(GMFlow, self).__init__()
        self.count=0
        self.num_scales = num_scales
        self.feature_channels = feature_channels
        self.upsample_factor = upsample_factor
        self.attention_type = attention_type
        self.num_transformer_layers = num_transformer_layers
        self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(1, -1, 1, 1))
        
        self.up_layerx2 = nn.Sequential(
            nn.Conv2d(256, 32, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )
        self.up_layerx4 = nn.Sequential(
            nn.Conv2d(256, 32, 3, 1, 1),
            nn.Upsample(scale_factor=4, mode='bilinear'),
        )
        # SAM Image Encoder
        self.sam = get_encoder_base(checkpoint=sam_checkpoint)
        self.sam.eval()
        for param in self.sam.parameters():
            param.requires_grad = False
        # CNN backbone
        self.backbone = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales)
        self.CFM = nn.Sequential(ResidualBlock(128+32, 128), ResidualBlock(128, 128))
        # Transformer
        self.transformer = FeatureTransformer(num_layers=num_transformer_layers,
                                              d_model=feature_channels,
                                              nhead=num_head,
                                              attention_type=attention_type,
                                              ffn_dim_expansion=ffn_dim_expansion,
                                              )


        # flow propagation with self-attn
        self.feature_flow_attn = FeatureFlowAttention(in_channels=feature_channels)

        # convex upsampling: concat feature0 and flow as input
        self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))

    def sam_preprocess(self, x):
        # normalize to N(0, 1)
        x = (x * 255.0 - self.pixel_mean.to(x.device)) / self.pixel_std.to(x.device)
        return x

    def extract_feature(self, img0, img1, viz=True, viz_dir="viz", cmap="turbo"):
        # 1) CNN backbone 特征（高→低 -> 反转为 低→高）
        concat = torch.cat((img0, img1), dim=0)            # [2B, C, H, W]
        features = self.backbone(concat)                   # list of [2B, C, H, W]
        features = features[::-1]

        feature0, feature1 = [], []
        for i in range(len(features)):
            f = features[i]
            a, b = torch.chunk(f, 2, dim=0)               # a->img0, b->img1
            feature0.append(a)                             # 低→高
            feature1.append(b)

        # 2) SAM encoder 语义特征
        img0_norm = self.sam_preprocess(img0)
        img1_norm = self.sam_preprocess(img1)
        sam_feature0 = self.sam(img0_norm)                 # 形如 (B, C_s, h, w)
        sam_feature1 = self.sam(img1_norm)

        # 3) 在融合前，先做“要可视化的三种张量”的快照（以两层为例）
        #    CNN-only
        cnn0_l0 = feature0[0]
        cnn0_l1 = feature0[1]
        cnn1_l0 = feature1[0]
        cnn1_l1 = feature1[1]

        #    SAM (上采样到对应层的空间分辨率)
        sam0_l0 = self.up_layerx2(sam_feature0)
        sam0_l1 = self.up_layerx4(sam_feature0)
        sam1_l0 = self.up_layerx2(sam_feature1)
        sam1_l1 = self.up_layerx4(sam_feature1)

        # 4) 融合
        fused0_l0 = self.CFM(torch.cat([cnn0_l0, sam0_l0], dim=1))
        fused0_l1 = self.CFM(torch.cat([cnn0_l1, sam0_l1], dim=1))
        fused1_l0 = self.CFM(torch.cat([cnn1_l0, sam1_l0], dim=1))
        fused1_l1 = self.CFM(torch.cat([cnn1_l1, sam1_l1], dim=1))

        # 覆盖回列表（若你的后续网络需要这些）
        feature0[0] = fused0_l0
        feature0[1] = fused0_l1
        feature1[0] = fused1_l0
        feature1[1] = fused1_l1

        # 5) 可视化（可选）
        if viz:
            H, W = img0.shape[-2], img0.shape[-1]
            # img0 分支
            visualize_mean_map(cnn0_l0, save_dir=viz_dir, tag="img0_cnn_l0", resize_to=(H, W), cmap_name=cmap, count=self.count)
            visualize_mean_map(sam0_l0, save_dir=viz_dir, tag="img0_sam_l0", resize_to=(H, W), cmap_name=cmap, count=self.count)
            visualize_mean_map(fused0_l0, save_dir=viz_dir, tag="img0_fused_l0", resize_to=(H, W), cmap_name=cmap, count=self.count)

        self.count = self.count + 1

        return feature0, feature1

    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8,
                      ):
        if bilinear:
            up_flow = F.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * upsample_factor

        else:
            # convex upsampling
            concat = torch.cat((flow, feature), dim=1)

            mask = self.upsampler(concat)
            b, flow_channel, h, w = flow.shape
            mask = mask.view(b, 1, 9, self.upsample_factor, self.upsample_factor, h, w)  # [B, 1, 9, K, K, H, W]
            mask = torch.softmax(mask, dim=2)

            up_flow = F.unfold(self.upsample_factor * flow, [3, 3], padding=1)
            up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]

            up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W]
            up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
            up_flow = up_flow.reshape(b, flow_channel, self.upsample_factor * h,
                                      self.upsample_factor * w)  # [B, 2, K*H, K*W]

        return up_flow

    def forward(self, img0, img1,
                attn_splits_list=None,
                corr_radius_list=None,
                prop_radius_list=None,
                pred_bidir_flow=False,
                init_flow=None,
                **kwargs,
                ):

        results_dict = {}
        flow_preds = []

        # resolution low to high， list = [1, 128, 32, 32]
        feature0_list, feature1_list = self.extract_feature(img0, img1)  # list of features

        flow = init_flow

        assert len(attn_splits_list) == len(corr_radius_list) == len(prop_radius_list) == self.num_scales

        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]

            if pred_bidir_flow and scale_idx > 0:
                # predicting bidirectional flow with refinement
                feature0, feature1 = torch.cat((feature0, feature1), dim=0), torch.cat((feature1, feature0), dim=0)

            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))

            if scale_idx > 0:
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2

            if flow is not None:
                flow = flow.detach()
                feature1 = flow_warp(feature1, flow)  # [B, C, H, W]

            attn_splits = attn_splits_list[scale_idx]
            corr_radius = corr_radius_list[scale_idx]
            prop_radius = prop_radius_list[scale_idx]

            # add position to features
            feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)

            # Transformer
            feature0, feature1 = self.transformer(feature0, feature1, attn_num_splits=attn_splits)
            # feature0, feature1 = self.DRDBs(feature0, feature1)

            # correlation and softmax
            if corr_radius == -1:  # global matching
                flow_pred = global_correlation_softmax(feature0, feature1, pred_bidir_flow)[0]
            else:  # local matching
                flow_pred = local_correlation_softmax(feature0, feature1, corr_radius)[0]

            # flow or residual flow
            flow = flow + flow_pred if flow is not None else flow_pred

            # upsample to the original resolution for supervison
            if self.training:  # only need to upsample intermediate flow predictions at training time
                flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor)
                flow_preds.append(flow_bilinear)

            # flow propagation with self-attnqqa
            if pred_bidir_flow and scale_idx == 0:
                feature0 = torch.cat((feature0, feature1), dim=0)  # [2*B, C, H, W] for propagation
            flow = self.feature_flow_attn(feature0, flow.detach(),
                                          local_window_attn=prop_radius > 0,
                                          local_window_radius=prop_radius)

            # bilinear upsampling at training time except the last one
            if self.training and scale_idx < self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0, bilinear=True, upsample_factor=upsample_factor)
                flow_preds.append(flow_up)

            if scale_idx == self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0)
                flow_preds.append(flow_up)

        results_dict.update({'flow_preds': flow_preds})

        return results_dict
