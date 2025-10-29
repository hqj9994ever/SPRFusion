import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import glob
import random
import cv2
from CLIP import clip
from itertools import cycle
from utils import utils_image as util

device = "cuda" if torch.cuda.is_available() else "cpu"
#load clip
model, preprocess = clip.load("ViT-B/32", device=device, download_root="./clip_model/")#ViT-B/32
for para in model.parameters():
	para.requires_grad = False

def random_gamma_correction(image, gamma_range=(0.8, 2.0)):
    """
    Apply random gamma correction to an image.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        gamma_range (tuple): Range of gamma values (min_gamma, max_gamma).

    Returns:
        numpy.ndarray: Image after random gamma correction.
    """

    assert gamma_range[0] > 0, "The minimum gamma value must be greater than 0"
    assert gamma_range[1] >= gamma_range[0], "The maximum gamma value must be greater than or equal to the minimum"

    gamma = random.uniform(gamma_range[0], gamma_range[1])

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")

    corrected_image = cv2.LUT(image, table)

    return corrected_image

def transform_matrix_offset_center(matrix, x, y):
	"""Return transform matrix offset center.

	Parameters
	----------
	matrix : numpy array
		Transform matrix
	x, y : int
		Size of image.

	Examples
	--------
	- See ``rotation``, ``shear``, ``zoom``.
	"""
	o_x = float(x) / 2 + 0.5
	o_y = float(y) / 2 + 0.5 
	offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
	reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
	transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
	return transform_matrix 

def img_rotate(img, angle, center=None, scale=1.0):
	"""Rotate image.
	Args:
		img (ndarray): Image to be rotated.
		angle (float): Rotation angle in degrees. Positive values mean
			counter-clockwise rotation.
		center (tuple[int]): Rotation center. If the center is None,
			initialize it as the center of the image. Default: None.
		scale (float): Isotropic scale factor. Default: 1.0.
	"""
	(h, w) = img.shape[:2]

	if center is None:
		center = (w // 2, h // 2)

	matrix = cv2.getRotationMatrix2D(center, angle, scale)
	rotated_img = cv2.warpAffine(img, matrix, (w, h),flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT, borderValue=(0,0,0),)
	return rotated_img

def zoom(x, zx, zy, row_axis=0, col_axis=1):
	zoom_matrix = np.array([[zx, 0, 0],
							[0, zy, 0],
							[0, 0, 1]])
	h, w = x.shape[row_axis], x.shape[col_axis]

	matrix = transform_matrix_offset_center(zoom_matrix, h, w) 
	x = cv2.warpAffine(x, matrix[:2, :], (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT, borderValue=(0, 0, 0))
	return x

def augmentation(img):
	hflip=random.random() < 0.5
	vflip=random.random() < 0.5
	rot90=random.random() < 0.5
	rot=random.random() < 0.3
	zo=random.random() < 0.3
	angle=random.random() * 180 - 90
	if hflip:
		img = cv2.flip(img, 1)
	if vflip:
		img = cv2.flip(img, 0)
	if rot90:
		img = img.transpose(1, 0, 2)
	if zo:
		zoom_range=(0.7, 1.3)
		zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
		img = zoom(img, zx, zy)
	if rot:
		img = img_rotate(img, angle)

	return img

def preprocess_aug(img, img_path):
	img = np.uint8((np.asarray(img)))
	img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
	img = augmentation(img)
	if "trainA" in img_path or "trainB" in img_path:
		img = random_gamma_correction(img)
	img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	return img


def populate_train_list(lowlight_images_path, overlight_images_path, normallight_images_path):
	
	image_list_lowlight = glob.glob(lowlight_images_path + "*")
	image_list_overlight = glob.glob(overlight_images_path + "*")
	image_list_normallight = glob.glob(normallight_images_path + "*")
	image_ref_list=image_list_normallight.copy()
	image_input_list=image_list_lowlight.copy() + image_list_overlight.copy()

	if len(image_list_normallight) == 0 or len(image_list_lowlight) == 0:
		raise Exception("one of the image lists is empty!", len(image_list_normallight),len(image_list_lowlight))
	
	if len(image_ref_list) < len(image_input_list) // 2:
		image_ref_list = list(next(cycle(image_ref_list)) for _ in range(len(image_input_list) // 2))
	elif len(image_input_list) < len(image_ref_list) * 2:
		image_input_list = list(next(cycle(image_input_list)) for _ in range(len(image_ref_list) * 2))
	
	train_list = image_input_list + image_ref_list
	random.shuffle(train_list)

	return train_list


class dataloader(Dataset):

	def __init__(self, lowlight_images_path, overlight_images_path, normallight_images_path):
		
		self.train_list = populate_train_list(lowlight_images_path, overlight_images_path, normallight_images_path)
		self.size = 224
		self.data_list = self.train_list
		print("Total training examples:", len(self.train_list))


	def __getitem__(self, index):

		data_path = self.data_list[index]
		
		data = Image.open(data_path)
		
		data = data.resize((self.size, self.size), Image.Resampling.LANCZOS)
		data = preprocess_aug(data, data_path)
		
		data = (np.asarray(data) / 255.0) 
		data = torch.from_numpy(data).float()
		image = data.permute(2, 0, 1).to(device).unsqueeze(0)
		image_features = model.encode_image(image)
		image_features /= image_features.norm(dim=-1, keepdim=True)

		if "trainC" in data_path:
			label = torch.from_numpy(np.array([0., 0., 1.]))
		elif "trainB" in data_path:
			label = torch.from_numpy(np.array([1., 0., 0.]))
		else:
			label = torch.from_numpy(np.array([0., 1., 0.]))
	
		return image_features, label

	def __len__(self):
		return len(self.data_list)


if __name__ == '__main__':
	img = util.imread_uint("test.png", n_channels=3)
	img = random_gamma_correction(img)
	util.imsave(img, "correction.png")
