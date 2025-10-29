from torch.utils.data import Dataset

import numpy as np
import glob
import random
import cv2

from utils import utils_image as util


random.seed(1143)


def populate_train_list(lowlight_images_path, overlight_images_path, normallight_images_path, flow_path):
	
	image_list_lowlight = glob.glob(lowlight_images_path + "*")
	image_list_overlight = glob.glob(overlight_images_path + "*")
	image_list_normallight = glob.glob(normallight_images_path + "*")
	flow_list_path = glob.glob(flow_path + "*") if flow_path is not None else None

	train_list1 = sorted(image_list_lowlight)
	train_list2 = sorted(image_list_overlight)
	train_list3 = sorted(image_list_normallight)
	train_list4 = sorted(flow_list_path) if flow_list_path is not None else None

	return train_list1, train_list2, train_list3, train_list4

	

class dataloader(Dataset):

	def __init__(self, lowlight_images_path, overlight_images_path, normallight_images_path, flow_path, patch_size, phase):

		self.train_list1, self.train_list2, self.train_list3, self.train_list4 = populate_train_list(lowlight_images_path, overlight_images_path, normallight_images_path, flow_path)
		self.phase = phase 
		self.patch_size = patch_size
		
		if phase == 'train':
			print("Total training examples:", len(self.train_list1))
		elif phase == 'val':
			print("Total validation examples:", len(self.train_list1))


	def __getitem__(self, index):
		if self.phase == 'train':
			data_lowlight_path = self.train_list1[index]
			data_overlight_path = self.train_list2[index]
			data_normallight_path = self.train_list3[index]
			data_flow_path = self.train_list4[index]
			data_lowlight = util.imread_uint(data_lowlight_path, n_channels=3)
			data_overlight = util.imread_uint(data_overlight_path, n_channels=3)
			data_normallight = util.imread_uint(data_normallight_path, n_channels=3)
			data_flow = np.load(data_flow_path)

			H, W = data_lowlight.shape[:2]
			# ---------------------------------
			# randomly crop the image (local)
			# ---------------------------------           
			rnd_h = random.randint(0, max(0, H - self.patch_size))
			rnd_w = random.randint(0, max(0, W - self.patch_size))
			patch_lowlight = data_lowlight[rnd_h:rnd_h+self.patch_size, rnd_w:rnd_w+self.patch_size, :]
			patch_overlight = data_overlight[rnd_h:rnd_h+self.patch_size, rnd_w:rnd_w+self.patch_size, :]
			patch_normallight = data_normallight[rnd_h:rnd_h+self.patch_size, rnd_w:rnd_w+self.patch_size, :]
			patch_flow = data_flow[rnd_h:rnd_h+self.patch_size, rnd_w:rnd_w+self.patch_size, :]
			# ---------------------------------
			# resize the image (global)
			# ---------------------------------  
			img_lowlight = cv2.resize(data_lowlight, (self.patch_size, self.patch_size), cv2.INTER_AREA)
			img_overlight = cv2.resize(data_overlight, (self.patch_size, self.patch_size), cv2.INTER_AREA)
            # ---------------------------------
            # augmentation - flip, rotate
            # ---------------------------------
			mode = [random.randint(0, 7), random.randint(0, 7)]
			patch_lowlight = util.augment_img(patch_lowlight, mode=mode[0])
			patch_overlight = util.augment_img(patch_overlight, mode=mode[0])
			patch_normallight = util.augment_img(patch_normallight, mode=mode[0])
			patch_flow = util.augment_flow(patch_flow, mode=mode[1])
			img_lowlight = util.augment_img(img_lowlight, mode=mode[0])
			img_overlight = util.augment_img(img_overlight, mode=mode[0])
			# ---------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # ---------------------------------
			patch_lowlight = util.uint2tensor3(patch_lowlight)
			patch_overlight = util.uint2tensor3(patch_overlight)
			patch_normallight = util.uint2tensor3(patch_normallight)
			patch_flow = util.single2tensor3(patch_flow)
			img_lowlight = util.uint2tensor3(img_lowlight)
			img_overlight = util.uint2tensor3(img_overlight)
			
			return patch_lowlight, patch_overlight, patch_normallight, patch_flow, img_lowlight, img_overlight, data_lowlight_path
		
		elif self.phase == 'val':
			data_lowlight_path = self.train_list1[index]
			data_overlight_path = self.train_list2[index]
			data_normallight_path = self.train_list3[index]
			data_lowlight = util.imread_uint(data_lowlight_path, n_channels=3)
			data_overlight = util.imread_uint(data_overlight_path, n_channels=3)
			data_normallight = util.imread_uint(data_normallight_path, n_channels=3)

			img_lowlight = util.modcrop(data_lowlight, scale=32)
			img_overlight = util.modcrop(data_overlight, scale=32)
			img_normallight = util.modcrop(data_normallight, scale=32)

			img_lowlight = util.uint2tensor3(img_lowlight)
			img_overlight = util.uint2tensor3(img_overlight)
			img_normallight = util.uint2tensor3(img_normallight)

			return img_lowlight, img_overlight, img_normallight, data_lowlight_path

	def __len__(self):
		return len(self.train_list1)
	
