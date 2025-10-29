"""
Visualize per-layer features of SAM ViT-B image encoder.

Usage:
  python visualize_sam_vitb_feats.py \
      --image path/to/your_image.jpg \
      --checkpoint sam_vit_b_01ec64.pth \
      --outdir outputs_sam_vitb \
      --layers all          # 也可指定: 0,4,8,12
      --mode both           # 可选: mean / pca / both
      --longest_side 1024   # SAM 默认
"""

import os
import math
import argparse
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide


# -------------------------
# Utilities
# -------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def to_uint8(img01: np.ndarray) -> np.ndarray:
    img = np.clip(img01, 0, 1)
    img = (img * 255.0 + 0.5).astype(np.uint8)
    return img

def save_gray(path, m01: np.ndarray):
    m01 = np.clip(m01, 0, 1)
    cv2.imwrite(path, (m01 * 255.0).astype(np.uint8))

def save_color(path, im01: np.ndarray):
    # RGB->BGR for OpenCV
    cv2.imwrite(path, (np.clip(im01, 0, 1)[:, :, ::-1] * 255.0).astype(np.uint8))

def tensor_to_hwC(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, H, W, C) -> (H, W, C) for B=1
    """
    assert x.ndim == 4 and x.shape[0] == 1
    B, H, W, C = x.shape
    return x[0].reshape(H, W, C)

def minmax01(arr: np.ndarray, eps=1e-8) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + eps)

def pca_to_rgb(feat_hwC: torch.Tensor, out_ch: int = 3) -> np.ndarray:
    """
    feat_hwC: (H, W, C) torch tensor
    Return: (H, W, 3) numpy in [0,1]
    """
    H, W, C = feat_hwC.shape
    X = feat_hwC.reshape(-1, C)  # (HW, C)
    X = X - X.mean(dim=0, keepdim=True)
    # SVD: X = U S Vt; principal components: V
    # Use float32 CPU for stability if needed
    device = X.device
    Xc = X.float()
    U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    V = Vt.T  # (C, C)
    comps = Xc @ V[:, :out_ch]  # (HW, 3)
    comps = comps.reshape(H, W, out_ch)
    comps = comps.detach().cpu().numpy()
    # normalize each channel to [0,1], then stack
    out = np.zeros_like(comps)
    for k in range(out_ch):
        out[..., k] = minmax01(comps[..., k])
    return out

def upsample_to(img01: np.ndarray, target_hw: tuple) -> np.ndarray:
    th, tw = target_hw
    return cv2.resize(img01, (tw, th), interpolation=cv2.INTER_CUBIC)

# -------------------------
# Hook helpers
# -------------------------
class FeatureCatcher:
    """
    Collects per-block outputs from SAM image encoder (ViT-B).
    We hook the output *tokens* right after each Transformer block.
    """
    def __init__(self, image_encoder):
        self.blocks = image_encoder.blocks  # list of Transformer blocks
        self.handles = []
        self.outputs: Dict[int, torch.Tensor] = {}

    def _hook(self, idx):
        def fn(_module, _inp, out):
            # 'out' here is token sequence after the block:
            # shape (B, HW, C) in SAM's ViT implementation (before neck)
            self.outputs[idx] = out.detach()
        return fn

    def register(self):
        self.clear()
        for i, blk in enumerate(self.blocks):
            h = blk.register_forward_hook(self._hook(i))
            self.handles.append(h)

    def clear(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []
        self.outputs.clear()

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Dataset/test_data/train/trainC/001.png")
    parser.add_argument("--checkpoint", type=str, required=True, help="model_zoo/ckpt/sam_vit_b_01ec64.pth")
    parser.add_argument("--outdir", type=str, default="sam_vitb_feats_out")
    parser.add_argument("--layers", type=str, default="all", help="Comma list (e.g., 0,4,8,11) or 'all'")
    parser.add_argument("--mode", type=str, default="both", choices=["mean", "pca", "both"])
    parser.add_argument("--longest_side", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    # 1) Load SAM ViT-B
    sam = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    sam.to(args.device)
    sam.eval()

    # 2) Prepare predictor + transforms (handles SAM's preprocessing / padding logic)
    predictor = SamPredictor(sam)
    img_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Keep original size for final upsampling
    H0, W0 = img_rgb.shape[:2]

    # 3) Register hooks on image encoder blocks
    catcher = FeatureCatcher(sam.image_encoder)
    catcher.register()

    # 4) Trigger forward by setting image (this runs the image encoder once)
    predictor.set_image(img_rgb)  # internally resizes to 'longest_side' (default 1024) and normalizes

    # 5) Decide which layers to export
    if args.layers.strip().lower() == "all":
        layer_indices = list(sorted(catcher.outputs.keys()))
    else:
        layer_indices = [int(x) for x in args.layers.split(",")]

    # 6) Grab tokens before neck (B, HW, C), reshape to (H, W, C)
    #    Also save final image embedding from encoder output (B, 256, 64, 64)
    #    predictor.features is (1, 256, 64, 64)
    final_embed = predictor.get_image_embedding().detach().cpu()  # (1, 256, 64, 64)
    Cfin, Hfin, Wfin = final_embed.shape[1:]

    # Save final embedding quicklooks
    if args.mode in ["mean", "both"]:
        mean_final = final_embed.mean(dim=1)[0].numpy()  # (64, 64)
        mean_final = minmax01(mean_final)
        mean_final_up = upsample_to(mean_final, (H0, W0))
        save_gray(os.path.join(args.outdir, f"final_embed_mean64x64_up.png"), mean_final_up)

    if args.mode in ["pca", "both"]:
        fin_hwC = final_embed.permute(0, 2, 3, 1)  # (1, 64, 64, 256)
        fin_rgb01 = pca_to_rgb(fin_hwC[0])
        fin_rgb_up = upsample_to(fin_rgb01, (H0, W0))
        save_color(os.path.join(args.outdir, f"final_embed_pca64x64_up.png"), fin_rgb_up)

    # 7) Per-layer tokens
    for idx in layer_indices:
        if idx not in catcher.outputs:
            print(f"[Warn] Layer {idx} not captured (skip).")
            continue
        tok = catcher.outputs[idx].to("cpu")  # (B, HW, C)
        hwC = tensor_to_hwC(tok)  # (H, W, C) on CPU, float32
        H, W, C = hwC.shape

        if args.mode in ["mean", "both"]:
            mean_map = hwC.mean(dim=-1).numpy()  # (H, W)
            mean_map = minmax01(mean_map)
            mean_up = upsample_to(mean_map, (H0, W0))
            save_gray(os.path.join(args.outdir, f"layer{idx:02d}_mean_{H}x{W}_up.png"), mean_up)

        if args.mode in ["pca", "both"]:
            rgb01 = pca_to_rgb(hwC)  # (H, W, 3) in [0,1]
            rgb_up = upsample_to(rgb01, (H0, W0))
            save_color(os.path.join(args.outdir, f"layer{idx:02d}_pca_{H}x{W}_up.png"), rgb_up)

    print(f"Done. Saved to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
