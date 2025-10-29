import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ---- helpers ----
def _to_mean_map(t: torch.Tensor) -> torch.Tensor:
    """
    t: (B,C,H,W) or (C,H,W)
    return: (B,1,H,W) mean map（按通道取均值）
    """
    if t.ndim == 3:
        t = t.unsqueeze(0)   # (1,C,H,W)
    assert t.ndim == 4, f"expect 4D, got {t.shape}"
    m = t.float().mean(dim=1, keepdim=True)  # (B,1,H,W)
    return m

def _minmax01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """逐张（per-sample）做 [0,1] 归一化，避免不同样本相互影响"""
    B = x.shape[0]
    x = x.clone()
    x_ = x.view(B, -1)
    mn = x_.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    mx = x_.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    return (x - mn) / (mx - mn + eps)

def _apply_matplotlib_cmap(mm01: torch.Tensor, cmap_name: str = "grey") -> np.ndarray:
    """
    mm01: (B,1,H,W) in [0,1]
    return: uint8 RGB (B,H,W,3)
    """
    B, _, H, W = mm01.shape
    cmap = cm.get_cmap(cmap_name)
    mm_np = mm01.squeeze(1).detach().cpu().numpy()  # (B,H,W)
    out = []
    for i in range(B):
        rgba = cmap(mm_np[i])           # (H,W,4), float in [0,1]
        rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
        out.append(rgb)
    return np.stack(out, axis=0)        # (B,H,W,3)

def visualize_mean_map(
    feat: torch.Tensor,
    save_dir: str = None,
    tag: str = "feat",
    resize_to: tuple = None,
    cmap_name: str = "magma",
    show: bool = False,
    count: int = 0,
):
    """
    feat: (B,C,H,W)/(C,H,W)
    resize_to: (H, W) 若给定则双线性缩放到该尺寸（例如与输入图像一致）
    """
    mm = _to_mean_map(feat)             # (B,1,H,W)
    if resize_to is not None:
        mm = F.interpolate(mm, size=resize_to, mode="bilinear", align_corners=False)
    mm01 = _minmax01(mm)
    rgb = _apply_matplotlib_cmap(mm01, cmap_name=cmap_name)  # (B,H,W,3) uint8

    save_dir = os.path.join(save_dir, str(count).zfill(3))
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for i in range(rgb.shape[0]):
            path = os.path.join(save_dir, f"{tag}_b{i}.png")
            plt.imsave(path, rgb[i])
    if show:
        # 简单展示第0张
        plt.figure()
        plt.title(tag)
        plt.axis('off')
        plt.imshow(rgb[0])
        plt.show()
    return rgb  # 返回可用于进一步处理或单元测试
