import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import torch
import logging
from collections import OrderedDict

from model.network import AttING, DualAttModule, FuseModule
from utils import utils_image as util
from utils import utils_logger


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='PyTorch implementation of TextMEF')
parser.add_argument('--input_u', help='directory of input folder', default='Dataset/test_data/SICE/trainA/')
parser.add_argument('--input_o', help='directory of input folder', default='Dataset/test_data/SICE/trainB/')
parser.add_argument('--gt', help='directory of gt folder', default='Dataset/test_data/SICE/trainC/')
parser.add_argument('--output', help='directory of output folder', default='inference/')
parser.add_argument('--model_A_path', help='test ckpt path', default='model_zoo/epoch_400_A_static_local+global.pth')
parser.add_argument('--model_D_path', help='test ckpt path', default='model_zoo/epoch_400_D_static_local+global.pth')
parser.add_argument('--model_G_path', help='test ckpt path', default='model_zoo/epoch_400_G_static_local+global.pth')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_A = AttING(in_channels=3, channels=64)
model_D = DualAttModule(in_dim=64)
model_G = FuseModule(in_channels=64)
model_A.load_state_dict(torch.load(args.model_A_path))
model_D.load_state_dict(torch.load(args.model_D_path))
model_G.load_state_dict(torch.load(args.model_G_path))
model_A.eval()
model_D.eval()
model_G.eval()
for k, v in model_A.named_parameters():
	v.requires_grad = False
for k, v in model_D.named_parameters():
	v.requires_grad = False
for k, v in model_G.named_parameters():
	v.requires_grad = False
model_A = model_A.to(device)
model_D = model_D.to(device)
model_G = model_G.to(device)

test_results = OrderedDict()
test_results['psnr'] = []
test_results['ssim'] = []

def inference(lowlight_image_path, overlight_image_path, normallight_image_path): 

	data_lowlight = util.imread_uint(lowlight_image_path, n_channels=3)
	data_lowlight = util.modcrop(data_lowlight, scale=32)
	data_lowlight = util.uint2tensor4(data_lowlight).to(device)
	
	data_overlight = util.imread_uint(overlight_image_path, n_channels=3)
	data_overlight = util.modcrop(data_overlight, scale=32)
	data_overlight = util.uint2tensor4(data_overlight).to(device)

	data_normallight = util.imread_uint(normallight_image_path, n_channels=3)
	data_normallight = util.modcrop(data_normallight, scale=32)

	identity_u, correct_u = model_A(data_lowlight)
	identity_o, correct_o = model_A(data_overlight)
	align_u, align_o = model_D(identity_u, identity_o)
	output = model_G(align_u, align_o)
	output = util.tensor2uint(output)
	psnr = util.calculate_psnr(output, data_normallight, border=0)
	ssim = util.calculate_ssim(output, data_normallight, border=0)
	test_results['psnr'].append(psnr)
	test_results['ssim'].append(ssim)

	logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(os.path.basename(lowlight_image_path), psnr, ssim))
	
	image_path = os.path.join(args.output, os.path.basename(lowlight_image_path))

	result_path = image_path
	util.imsave(output, result_path)


if __name__ == '__main__':
	os.makedirs(args.output, exist_ok=True)

	logger_name = 'test'
	utils_logger.logger_info(logger_name, os.path.join(args.output, logger_name+'.log'))
	logger = logging.getLogger(logger_name)
	
	with torch.no_grad():
		filePath_u = args.input_u
		filePath_o = args.input_o
		filePath_gt = args.gt
		file_list_u = sorted(os.listdir(filePath_u))
		file_list_o = sorted(os.listdir(filePath_o))
		file_list_gt = sorted(os.listdir(filePath_gt))
  
		for file_name_u, file_name_o, file_name_gt in zip(file_list_u, file_list_o, file_list_gt):
			image_u_path = os.path.join(filePath_u, file_name_u)
			image_o_path = os.path.join(filePath_o, file_name_o)
			image_gt_path = os.path.join(filePath_gt, file_name_gt)
			inference(image_u_path, image_o_path, image_gt_path)

		ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
		ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
		logger.info('Average PSNR/SSIM(RGB) - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(ave_psnr, ave_ssim))
		

