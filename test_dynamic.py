import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import torch
import logging
from collections import OrderedDict

from model.network import AttING, align_FG, DualAttModule, FuseModule
from utils import utils_image as util
from utils import utils_logger
import time


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='PyTorch implementation of TextMEF')
parser.add_argument('--input_u', help='directory of input folder', default='Dataset/SICE1024/')
parser.add_argument('--input_o', help='directory of input folder', default='Dataset/SICE1024/')
parser.add_argument('--gt', help='directory of gt folder', default='Dataset/test_data/SICE/trainC/')
parser.add_argument('--output', help='directory of output folder', default='inference/')
parser.add_argument('--use_align', help='use align or not', action='store_true', default=False)
parser.add_argument('--need_H', help='have ground truth or not', action='store_true', default=False)
parser.add_argument('--model_A_path', help='test ckpt path', default='model_zoo/sam+psp-ff+global+local/epoch_500_A.pth')
parser.add_argument('--model_F_path', help='test ckpt path', default='model_zoo/sam+psp-ff+global+local/epoch_500_F.pth')
parser.add_argument('--model_D_path', help='test ckpt path', default='model_zoo/sam+psp-ff+global+local/epoch_500_D.pth')
parser.add_argument('--model_G_path', help='test ckpt path', default='model_zoo/sam+psp-ff+global+local/epoch_500_G.pth')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_A = AttING(in_channels=3, channels=64)
model_F = align_FG()
model_D = DualAttModule(in_dim=64)
model_G = FuseModule(in_channels=64)
model_A.load_state_dict(torch.load(args.model_A_path))
model_F.load_state_dict(torch.load(args.model_F_path))
model_D.load_state_dict(torch.load(args.model_D_path))
model_G.load_state_dict(torch.load(args.model_G_path))
model_A.eval()
model_F.eval()
model_D.eval()
model_G.eval()
for k, v in model_A.named_parameters():
	v.requires_grad = False
for k, v in model_F.named_parameters():
	v.requires_grad = False
for k, v in model_D.named_parameters():
	v.requires_grad = False
for k, v in model_G.named_parameters():
	v.requires_grad = False
model_A = model_A.to(device)
model_F = model_F.to(device)
model_D = model_D.to(device)
model_G = model_G.to(device)

test_results = OrderedDict()
test_results['psnr'] = []
test_results['ssim'] = []


def get_padded_size(h, w):
	new_h = ((h - 1) // 32 + 1) * 32
	new_w = ((w - 1) // 32 + 1) * 32
	return new_h, new_w


def inference(lowlight_image_path, overlight_image_path, normallight_image_path=None):
	
	data_lowlight = util.imread_uint(lowlight_image_path, n_channels=3)
	h, w = data_lowlight.shape[:2]
	
	data_overlight = util.imread_uint(overlight_image_path, n_channels=3)
	
	if args.need_H:
		data_normallight = util.imread_uint(normallight_image_path, n_channels=3)

	new_h, new_w = get_padded_size(h, w)
	
	data_lowlight = util.uint2tensor4(util.imresize(data_lowlight, (new_h, new_w))).to(device)
	data_overlight = util.uint2tensor4(util.imresize(data_overlight, (new_h, new_w))).to(device)
	
	identity_u, correct_u = model_A(data_lowlight)
	identity_o, correct_o = model_A(data_overlight)
	# util.imsave(util.tensor2uint(correct_o), 'over.png')
	# util.imsave(util.tensor2uint(correct_u), 'under.png')

	if args.use_align == True:
		align_o, f_w, f_s = model_F(correct_o, correct_u, identity_o)
		# align_o, f_w, f_s = model_F(data_overlight, data_lowlight, identity_o)
		align_u, align_o = model_D(identity_u, align_o)
	else:
		align_u, align_o = model_D(identity_u, identity_o)

	output = model_G(align_u, align_o)
	
	
	output = util.tensor2uint(output)
	output = util.imresize(output, (h, w))

	if args.need_H:
		psnr = util.calculate_psnr(output, data_normallight, border=0)
		ssim = util.calculate_ssim(output, data_normallight, border=0)
		test_results['psnr'].append(psnr)
		test_results['ssim'].append(ssim)
		logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(os.path.basename(lowlight_image_path), psnr, ssim))
	else:
		logger.info('{:s}.'.format(os.path.basename(lowlight_image_path)))
	
	image_path = os.path.join(args.output, os.path.basename(lowlight_image_path))
	util.imsave(output, image_path)



if __name__ == '__main__':

	os.makedirs(args.output, exist_ok=True)

	logger_name = 'test'
	utils_logger.logger_info(logger_name, os.path.join(args.output, logger_name+'.log'))
	logger = logging.getLogger(logger_name)
	

	with torch.no_grad():
		filePath_u = args.input_u
		filePath_o = args.input_o
		file_list_u = sorted(os.listdir(filePath_u))
		file_list_o = sorted(os.listdir(filePath_o))

		if args.need_H:
			filePath_gt = args.gt
			file_list_gt = sorted(os.listdir(filePath_gt))
  
			for file_name_u, file_name_o, file_name_gt in zip(file_list_u, file_list_o, file_list_gt):
				image_u_path = os.path.join(filePath_u, file_name_u)
				image_o_path = os.path.join(filePath_o, file_name_o)
				image_gt_path = os.path.join(filePath_gt, file_name_gt)
				inference(image_u_path, image_o_path, image_gt_path)
		else:
			for file_name_u, file_name_o in zip(file_list_u, file_list_o):
				image_u_path = os.path.join(filePath_u, file_name_u)
				image_o_path = os.path.join(filePath_o, file_name_o)
				inference(image_u_path, image_o_path)

		if args.need_H:
			ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
			ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
			logger.info('Average PSNR/SSIM(RGB) - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(ave_psnr, ave_ssim))


		
	
		

