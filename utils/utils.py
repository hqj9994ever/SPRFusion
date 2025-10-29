import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import random
import numpy as np
import logging
from einops import rearrange
import torch.distributed as dist
import torch.nn.functional as F
import kornia
from kornia.filters.kernels import get_gaussian_kernel2d

from utils import utils_image as util



def to_3d(x):
	return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
	return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


def weights_init(m):
	classname = m.__class__.__name__ 
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)


def Get_color_MEF_tensor(cb1, cr1, cb2, cr2):

	cb1s = [x.clone().squeeze().cpu().numpy()*255 for x in torch.split(cb1, 1, dim=0)]
	cr1s = [x.clone().squeeze().cpu().numpy()*255 for x in torch.split(cr1, 1, dim=0)]
	cb2s = [x.clone().squeeze().cpu().numpy()*255 for x in torch.split(cb2, 1, dim=0)]
	cr2s = [x.clone().squeeze().cpu().numpy()*255 for x in torch.split(cr2, 1, dim=0)]
	cb = []
	cr = []
	for Cb1, Cr1, Cb2, Cr2 in zip(cb1s, cr1s, cb2s, cr2s):
		H, W = Cb1.shape
		Cb = np.ones((H, W))
		Cr = np.ones((H, W))
		for k in range(H):
			for n in range(W):
				if abs(Cb1[k, n] - 128) == 0 and abs(Cb2[k, n] - 128) == 0:
					Cb[k, n] = 128
				else:
					middle_1 = Cb1[k, n] * abs(Cb1[k, n] - 128) + Cb2[k, n] * abs(Cb2[k, n] - 128)
					middle_2 = abs(Cb1[k, n] - 128) + abs(Cb2[k, n] - 128)
					Cb[k, n] = middle_1 / middle_2
				if abs(Cr1[k, n] - 128) == 0 and abs(Cr2[k, n] - 128) == 0:
					Cr[k, n] = 128
				else:
					middle_3 = Cr1[k, n] * abs(Cr1[k, n] - 128) + Cr2[k, n] * abs(Cr2[k, n] - 128)
					middle_4 = abs(Cr1[k, n] - 128) + abs(Cr2[k, n] - 128)
					Cr[k, n] = middle_3 / middle_4
	cb.append(transforms.ToTensor()(Cb.astype(np.uint8)).reshape(1, 1, H, W))
	cr.append(transforms.ToTensor()(Cr.astype(np.uint8)).reshape(1, 1, H, W))
	cb = torch.cat(cb, dim=0).cuda()
	cr = torch.cat(cr, dim=0).cuda()
	return cb, cr


def YCrCb2RGB(input_im):
	im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
	mat = torch.tensor(
		[[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
	).cuda()
	bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
	temp = (im_flat + bias).mm(mat).cuda()
	out = (
		temp.reshape(
			list(input_im.size())[0],
			list(input_im.size())[2],
			list(input_im.size())[3],
			3,
		)
			.transpose(1, 3)
			.transpose(2, 3)
	)
	return out
				

def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)


def get_dist_info():
	if dist.is_available():
		initialized = dist.is_initialized()
	else:
		initialized = False
	if initialized:
		rank = dist.get_rank()
		world_size = dist.get_world_size()
	else:
		rank = 0
		world_size = 1
	return rank, world_size


def get_root_logger(logger_name='basicsr',
					log_level=logging.INFO,
					log_file=None):
	"""Get the root logger.

	The logger will be initialized if it has not been initialized. By default a
	StreamHandler will be added. If `log_file` is specified, a FileHandler will
	also be added.

	Args:
		logger_name (str): root logger name. Default: 'basicsr'.
		log_file (str | None): The log filename. If specified, a FileHandler
			will be added to the root logger.
		log_level (int): The root logger level. Note that only the process of
			rank 0 is affected, while other processes will set the level to
			"Error" and be silent most of the time.

	Returns:
		logging.Logger: The root logger.
	"""
	logger = logging.getLogger(logger_name)
	# if the logger has been initialized, just return it
	if logger.hasHandlers():
		return logger

	format_str = '%(asctime)s %(levelname)s: %(message)s'
	logging.basicConfig(format=format_str, level=log_level)
	rank, _ = get_dist_info()
	if rank != 0:
		logger.setLevel('ERROR')
	elif log_file is not None:
		file_handler = logging.FileHandler(log_file, 'w')
		file_handler.setFormatter(logging.Formatter(format_str))
		file_handler.setLevel(log_level)
		logger.addHandler(file_handler)

	return logger

	

def flow_warp2(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True, use_pad_mask=False):
	"""Warp an image or feature map with optical flow.

	Args:
		x (Tensor): Tensor with size (n, c, h, w).
		flow (Tensor): Tensor with size (n, h, w, 2), normal value.
		interp_mode (str): 'nearest' or 'bilinear' or 'nearest4'. Default: 'bilinear'.
		padding_mode (str): 'zeros' or 'border' or 'reflection'.
			Default: 'zeros'.
		align_corners (bool): Before pytorch 1.3, the default value is
			align_corners=True. After pytorch 1.3, the default value is
			align_corners=False. Here, we use the True as default.
		use_pad_mask (bool): only used for PWCNet, x is first padded with ones along the channel dimension.
			The mask is generated according to the grid_sample results of the padded dimension.


	Returns:
		Tensor: Warped image or feature map.
	"""
	# haodeh assert x.size()[-2:] == flow.size()[1:3] # temporaily turned off for image-wise shift
	n, _, h, w = x.size()
	x = x.float()
	# create mesh grid
	# grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x)) # an illegal memory access on TITAN RTX + PyTorch1.9.1
	grid_y, grid_x = torch.meshgrid(torch.arange(0, h, dtype=x.dtype, device=x.device),
									torch.arange(0, w, dtype=x.dtype, device=x.device))
	grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
	grid.requires_grad = False
	grid = grid.type_as(x)
	vgrid = grid + flow

	# if use_pad_mask: # for PWCNet
	#     x = F.pad(x, (0,0,0,0,0,1), mode='constant', value=1)

	# scale grid to [-1,1]
	if interp_mode == 'nearest4':  # todo: bug, no gradient for flow model in this case!!! but the result is good
		vgrid_x_floor = 2.0 * torch.floor(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
		vgrid_x_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
		vgrid_y_floor = 2.0 * torch.floor(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0
		vgrid_y_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0

		output00 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_floor), dim=3), mode='nearest',
								 padding_mode=padding_mode, align_corners=align_corners)
		output01 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_ceil), dim=3), mode='nearest',
								 padding_mode=padding_mode, align_corners=align_corners)
		output10 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_floor), dim=3), mode='nearest',
								 padding_mode=padding_mode, align_corners=align_corners)
		output11 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_ceil), dim=3), mode='nearest',
								 padding_mode=padding_mode, align_corners=align_corners)

		return torch.cat([output00, output01, output10, output11], 1)

	else:
		vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
		vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
		vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
		output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode,
							   align_corners=align_corners)

		# if use_pad_mask: # for PWCNet
		#     output = _flow_warp_masking(output)

		# TODO, what if align_corners=False
		return output
	

class AffineTransform(nn.Module):
	"""
	Add random affine transforms to a tensor image.
	Most functions are obtained from Kornia, difference:
	- gain the disp grid
	- no p and same_on_batch
	"""

	def __init__(self, degrees=0, translate=0.1):
		super(AffineTransform, self).__init__()
		self.trs = kornia.augmentation.RandomAffine(degrees, (translate, translate), return_transform=True, p=1)

	def forward(self, input):
		# image shape
		batch_size, _, height, weight = input.shape
		# affine transform
		warped, affine_param = self.trs(input)  # [batch_size, 3, 3]
		affine_theta = self.param_to_theta(affine_param, weight, height)  # [batch_size, 2, 3]
		# base + disp = grid -> disp = grid - base
		base = kornia.utils.create_meshgrid(height, weight, device=input.device).to(input.dtype)
		grid = F.affine_grid(affine_theta, size=input.size(), align_corners=False)  # [batch_size, height, weight, 2]
		disp = grid - base
		return warped, -disp

	@staticmethod
	def param_to_theta(param, weight, height):
		"""
		Convert affine transform matrix to theta in F.affine_grid
		:param param: affine transform matrix [batch_size, 3, 3]
		:param weight: image weight
		:param height: image height
		:return: theta in F.affine_grid [batch_size, 2, 3]
		"""

		theta = torch.zeros(size=(param.shape[0], 2, 3)).to(param.device)  # [batch_size, 2, 3]

		theta[:, 0, 0] = param[:, 0, 0]
		theta[:, 0, 1] = param[:, 0, 1] * height / weight
		theta[:, 0, 2] = param[:, 0, 2] * 2 / weight + param[:, 0, 0] + param[:, 0, 1] - 1
		theta[:, 1, 0] = param[:, 1, 0] * weight / height
		theta[:, 1, 1] = param[:, 1, 1]
		theta[:, 1, 2] = param[:, 1, 2] * 2 / height + param[:, 1, 0] + param[:, 1, 1] - 1

		return theta


class ElasticTransform(nn.Module):
	"""
	Add random elastic transforms to a tensor image.
	Most functions are obtained from Kornia, difference:
	- gain the disp grid
	- no p and same_on_batch
	"""

	def __init__(self, kernel_size: int = 63, sigma: float = 32, align_corners: bool = False, mode: str = "bilinear"):
		super(ElasticTransform, self).__init__()
		self.kernel_size = kernel_size
		self.sigma = sigma
		self.align_corners = align_corners
		self.mode = mode

	def forward(self, input):
		# generate noise
		batch_size, _, height, weight = input.shape
		noise = torch.rand(batch_size, 2, height, weight) * 2 - 1
		# elastic transform
		warped, disp = self.elastic_transform2d(input, noise)
		return warped, disp

	def elastic_transform2d(self, image: torch.Tensor, noise: torch.Tensor):
		if not isinstance(image, torch.Tensor):
			raise TypeError(f"Input image is not torch.Tensor. Got {type(image)}")

		if not isinstance(noise, torch.Tensor):
			raise TypeError(f"Input noise is not torch.Tensor. Got {type(noise)}")

		if not len(image.shape) == 4:
			raise ValueError(f"Invalid image shape, we expect BxCxHxW. Got: {image.shape}")

		if not len(noise.shape) == 4 or noise.shape[1] != 2:
			raise ValueError(f"Invalid noise shape, we expect Bx2xHxW. Got: {noise.shape}")

		# unpack hyper parameters
		kernel_size = self.kernel_size
		sigma = self.sigma
		align_corners = self.align_corners
		mode = self.mode
		device = image.device

		# Get Gaussian kernel for 'y' and 'x' displacement
		kernel_x: torch.Tensor = get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))[None]
		kernel_y: torch.Tensor = get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))[None]

		# Convolve over a random displacement matrix and scale them with 'alpha'
		disp_x: torch.Tensor = noise[:, :1].to(device)
		disp_y: torch.Tensor = noise[:, 1:].to(device)

		disp_x = kornia.filters.filter2d(disp_x, kernel=kernel_y, border_type="constant")
		disp_y = kornia.filters.filter2d(disp_y, kernel=kernel_x, border_type="constant")

		# stack and normalize displacement
		disp = torch.cat([disp_x, disp_y], dim=1).permute(0, 2, 3, 1)

		# Warp image based on displacement matrix
		b, c, h, w = image.shape
		grid = kornia.utils.create_meshgrid(h, w, device=image.device).to(image.dtype)
		warped = F.grid_sample(image, (grid + disp).clamp(-1, 1), align_corners=align_corners, mode=mode)

		return warped, disp



