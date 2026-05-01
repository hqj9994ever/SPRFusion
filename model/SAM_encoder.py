import torch
from model.SAM.image_encoder import ImageEncoderViT
from model.SAM.tiny_vit_sam import TinyViT
# from SAM.setup_mobile_sam import setup_model
from functools import partial


def get_encoder(checkpoint = None, ft_ckpt = False):
    encoder = ImageEncoderViT(depth=32,
            embed_dim=1280,
            img_size=1024,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=16,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[7, 15, 23, 31],
            window_size=14,
            out_chans=256,
            pos_crop_v0 = ft_ckpt)
    
    encoder.eval()
    
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location='cpu')
        new_state_dict = encoder.state_dict()
        for k in new_state_dict.keys():
            full_k = 'image_encoder.' + k
            new_state_dict[k] = state_dict[full_k]
        encoder.load_state_dict(new_state_dict)

    return encoder

def get_encoder_base(checkpoint = None):
    encoder = ImageEncoderViT(depth=12,
            embed_dim=768,
            img_size=1024,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=256,)
    
    encoder.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location='cpu')
        new_state_dict = encoder.state_dict()
        for k in new_state_dict.keys():
            full_k = 'image_encoder.' + k
            new_state_dict[k] = state_dict[full_k]
        encoder.load_state_dict(new_state_dict)

    return encoder


def get_encoder_tiny(checkpoint=None, device='cpu'):
    model = TinyViT(
        img_size=1024,   # 这里只作为初始化占位，不在 forward 中写死
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8,
    )

    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location='cpu')

        if 'model' in ckpt:
            ckpt = ckpt['model']
        elif 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']

        # 如果加载的是完整 mobile_sam 权重，只取 image_encoder 部分
        cleaned = {}
        for k, v in ckpt.items():
            if k.startswith('image_encoder.'):
                cleaned[k[len('image_encoder.'):]] = v
            elif not k.startswith('prompt_encoder') and not k.startswith('mask_decoder'):
                cleaned[k] = v

        msg = model.load_state_dict(cleaned, strict=False)
        print('[TinyViT] Missing keys:', msg.missing_keys)
        print('[TinyViT] Unexpected keys:', msg.unexpected_keys)

    model.eval()
    return model.to(device)



import os
import math
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def minmax_norm(x, eps=1e-8):
    x = x - x.min()
    x = x / (x.max() + eps)
    return x


def load_image_as_tensor(img_path, device='cpu'):
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)

    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0).to(device)   # [1, 3, H, W]

    return img_np, img_tensor


def save_input_image(img_np, save_path):
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_bgr)


def save_feature_mean_map(feat, save_path):
    """
    feat: [B, C, H, W]
    """
    fmap = feat[0].mean(dim=0).detach().cpu().numpy()
    fmap = minmax_norm(fmap)
    fmap = (fmap * 255).astype(np.uint8)
    cv2.imwrite(save_path, fmap)


def save_feature_pca_map(feat, save_path):
    """
    feat: [B, C, H, W]
    PCA压缩成3通道伪彩色图
    """
    x = feat[0].detach().cpu()   # [C, H, W]
    C, H, W = x.shape

    x = x.reshape(C, -1).transpose(0, 1)   # [H*W, C]
    x = x - x.mean(dim=0, keepdim=True)

    U, S, V = torch.pca_lowrank(x, q=3)
    x_pca = x @ V[:, :3]
    x_pca = x_pca.reshape(H, W, 3).numpy()

    out = np.zeros_like(x_pca)
    for i in range(3):
        out[..., i] = minmax_norm(x_pca[..., i])

    out = (out * 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, out)


def save_feature_channel_grid(feat, save_path, max_channels=16):
    """
    feat: [B, C, H, W]
    """
    x = feat[0].detach().cpu()   # [C, H, W]
    C, H, W = x.shape
    n = min(C, max_channels)

    cols = int(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    for i in range(rows * cols):
        r = i // cols
        c = i % cols
        ax = axes[r, c]
        ax.axis("off")

        if i < n:
            ch = x[i].numpy()
            ch = minmax_norm(ch)
            ax.imshow(ch, cmap='viridis')
            ax.set_title(f'ch{i}', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_overlay_heatmap(img_np, feat, save_path):
    """
    把特征均值图上采样后叠加到原图上
    img_np: [H, W, 3], RGB
    feat:   [B, C, h, w]
    """
    H, W = img_np.shape[:2]

    fmap = feat[0].mean(dim=0).detach().cpu().numpy()
    fmap = minmax_norm(fmap)
    fmap = (fmap * 255).astype(np.uint8)
    fmap = cv2.resize(fmap, (W, H), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(fmap, cv2.COLORMAP_JET)

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.5, heatmap, 0.5, 0)
    cv2.imwrite(save_path, overlay)


def visualize_one_image(encoder, img_path, save_dir, device='cpu'):
    ensure_dir(save_dir)

    img_np, img_tensor = load_image_as_tensor(img_path, device=device)

    encoder.eval()
    with torch.no_grad():
        outputs = encoder(img_tensor)

    # 兼容两种情况：
    # 1) encoder 返回一个特征
    # 2) encoder 返回多个特征
    if isinstance(outputs, (tuple, list)):
        feats = list(outputs)
    else:
        feats = [outputs]

    save_input_image(img_np, os.path.join(save_dir, 'input.png'))

    for i, feat in enumerate(feats):
        if feat is None:
            continue

        print(f'feat{i+1}.shape = {feat.shape}')

        save_feature_mean_map(
            feat, os.path.join(save_dir, f'feat{i+1}_mean.png')
        )
        save_feature_pca_map(
            feat, os.path.join(save_dir, f'feat{i+1}_pca.png')
        )
        save_feature_channel_grid(
            feat, os.path.join(save_dir, f'feat{i+1}_channels.png'), max_channels=16
        )
        save_overlay_heatmap(
            img_np, feat, os.path.join(save_dir, f'feat{i+1}_overlay.png')
        )

    print(f'visualization saved to: {save_dir}')



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder = get_encoder_tiny(
        checkpoint='/home/test/code/pytorch_hqj/SPRFusion/model_zoo/ckpt/mobile_sam.pt'
    ).to(device)

    img_path = '/home/test/code/pytorch_hqj/SPRFusion/Dataset/test/under.JPG'
    save_dir = './feat_vis_single'

    visualize_one_image(
        encoder=encoder,
        img_path=img_path,
        save_dir=save_dir,
        device=device
    )