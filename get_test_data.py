import argparse
import pathlib
import warnings

import os
os.environ['CUDA_VISIBLE_DIVICES'] = '7'
import cv2
import numpy as np
import kornia
import torch.backends.cudnn
import torch.cuda
import torch.utils.data
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from typing import List

from utils.utils import AffineTransform
from utils.utils import ElasticTransform
from utils.utils import flow_warp2
from utils import utils_image as util


class getDeformableImages:
    """
    principle: ir -> ir_warp
    """
    def __init__(self):
        # hardware settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # deformable transforms
        self.elastic = ElasticTransform(kernel_size=301, sigma=32)
        self.affine  = AffineTransform(degrees=0, translate=0.03)

    @torch.no_grad()
    def __call__(self, under_folder: pathlib.Path, over_folder: pathlib.Path, dst: pathlib.Path):

        # get images list
        under_list = [x for x in sorted(under_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        over_list = [x for x in sorted(over_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

        # starting generate deformable infrared image
        loader = tqdm(zip(under_list, over_list))
        for under_path, over_path in loader:
            name = under_path.name
            loader.set_description(f'warp: {name}')
            name_disp = name.split('.')[0] + '_disp.npy'

            # read images
            under = self.imread(under_path, unsqueeze=True)
            over = self.imread(over_path, unsqueeze=True)
            under = under.to(self.device)
            over = over.to(self.device)

            # get deformable images
            over_affine, affine_disp = self.affine(over)
            over_elastic, elastic_disp = self.elastic(over_affine)
            disp = affine_disp + elastic_disp  # cumulative disp grid [batch_size, height, weight, 2]
            over_warp = over_elastic

            _, _, h, w = over_warp.shape
            grid = kornia.utils.create_meshgrid(h, w, device=over_warp.device).to(over_warp.dtype)
            new_grid = grid + disp

            flow = torch.zeros_like(disp)
            flow[:, :, :, 0] = (grid[:, :, :, 0] + disp[:, :, :, 0] + 1.0) * max(w - 1, 1) / 2.0
            flow[:, :, :, 1] = (grid[:, :, :, 1] + disp[:, :, :, 1] + 1.0) * max(h - 1, 1) / 2.0
            grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
            grid2 = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
            grid2 = grid2.unsqueeze(0).to(self.device)
            flow = flow - grid2

            # under_warp = flow_warp2(under, flow)

            # draw grid
            img_grid = self._draw_grid(over.squeeze().cpu().numpy(), 24).to(self.device)

            # new_grid = new_grid.permute(0, 2, 3, 1)
            warp_grid = torch.nn.functional.grid_sample(img_grid.unsqueeze(0), new_grid, padding_mode='zeros', align_corners=False)
            # raw image w/o warp
            over_raw_grid  = 0.8 * over + 0.2 * img_grid
            over_raw_grid  = torch.clamp(over_raw_grid, 0, 1)
            # warped grid & warped ir image
            over_warp_grid = 0.8 * over_warp + 0.2 * warp_grid
            over_warp_grid = torch.clamp(over_warp_grid, 0, 1)
            # disp
            disp_npy = flow.data.squeeze().cpu().numpy()

            # save disp
            if not os.path.exists(dst):
                os.makedirs(dst)
            os.makedirs(dst / 'flow', exist_ok=True)
            np.save(dst / 'flow' / name_disp, disp_npy)
            # save deformable images
            self.imsave(under, dst / 'under', name)
            self.imsave(over_warp, dst / 'over_warp', name)            
            # self.imsave(under_warp, dst / 'under_warp', name)
            # self.imsave(0.6*over_warp+0.4*under_warp, dst / 'fuse', name)
            self.imsave(warp_grid, dst / 'warp_grid', name)
            self.imsave(over_warp_grid, dst / 'over_warp_grid', name)
            # self.imsave(over_raw_grid, dst / 'over_raw_grid', name)


    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_UNCHANGED, unsqueeze=False):
        im_cv = cv2.imread(str(path), flags)
        im_cv = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = torch.from_numpy(np.ascontiguousarray(im_cv)).permute(2, 0, 1).float().div(255.)
        return im_ts.unsqueeze(0) if unsqueeze else im_ts

    @staticmethod
    def imsave(im_s: List[Tensor], dst: pathlib.Path, im_name: str = ''):
        """
        save images to path
        :param im_s: image(s)
        :param dst: if one image: path; if multiple images: folder path
        :param im_name: name of image
        """

        im_s = im_s if type(im_s) == list else [im_s]
        dst = [dst / str(i + 1).zfill(3) / im_name for i in range(len(im_s))] if len(im_s) != 1 else [dst / im_name]
        for im_ts, p in zip(im_s, dst):
            im_ts = im_ts.squeeze().cpu()
            p.parent.mkdir(parents=True, exist_ok=True)
            im_cv = im_ts.data.squeeze().float().clamp_(0, 1).cpu().numpy()
            im_cv = np.transpose(im_cv, (1, 2, 0))
            im_cv = np.uint8((im_cv*255.0).round())
            im_cv = im_cv[:, :, [2, 1, 0]]
            cv2.imwrite(str(p), im_cv)

    @staticmethod
    def _draw_grid(im_cv, grid_size: int = 10):
        im_gd_cv = np.full_like(im_cv, 255.0)
        im_gd_cv = np.transpose(im_gd_cv, (1, 2, 0))
        color = (255, 0, 0)
        
        height, width = im_cv.shape[1:]
        for x in range(0, width - 1, grid_size):
            cv2.line(im_gd_cv, (x, 0), (x, height), color, 1, 1)
        for y in range(0, height - 1, grid_size):
            cv2.line(im_gd_cv, (0, y), (width, y), color, 1, 1)

        im_gd_ts = torch.from_numpy(np.ascontiguousarray(im_gd_cv)).permute(2, 0, 1).float().div(255.)
        return im_gd_ts
    

def hyper_args():
    """
    get hyper parameters from args
    """

    parser = argparse.ArgumentParser(description='Generating deformable testing data')
    # dataset
    parser.add_argument('--under', default='dataset/SICE/trainA/', type=pathlib.Path)
    parser.add_argument('--over', default='dataset/SICE/trainB/', type=pathlib.Path)
    parser.add_argument('--dst', default='dataset/SICE/', help='fuse image save folder', type=pathlib.Path)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    warnings.filterwarnings("ignore")
    args = hyper_args()
    data = getDeformableImages()
    data(under_folder=args.under, over_folder=args.over, dst=args.dst)
