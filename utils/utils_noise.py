# We adopt the same noise sampling procedure as in "Unprocessing Images for Learned Raw Denoising" by Brooks et al. CVPR2019

import torch
import torch.distributions as dist
import numpy as np


def random_noise_levels_dnd():
    """If the target dataset is DND, use this function 
    Generates random noise levels from a log-log linear distribution."""
    log_min_shot_noise = torch.log10(torch.Tensor([0.0001]))
    #log_max_shot_noise = torch.log10(torch.Tensor([0.012]))
    log_max_shot_noise = torch.log10(torch.Tensor([0.012]))
    distribution = dist.uniform.Uniform(log_min_shot_noise, log_max_shot_noise)
    log_shot_noise = distribution.sample()
    shot_noise = torch.pow(10, log_shot_noise)
    distribution = dist.normal.Normal(torch.Tensor([0.0]), torch.Tensor([0.25]))
    read_noise = distribution.sample().clamp(min=-1.5, max=1.5)
    #line = lambda x: 2.18 * x + 1.20
    line = lambda x: 2.275 * x + 1.47
    log_read_noise = line(log_shot_noise) + read_noise
    read_noise = torch.pow(10, log_read_noise)
    return shot_noise, read_noise


def random_noise_levels_sidd():
    """ If the target dataset is SIDD, use this function
    Where read_noise in SIDD is not 0 """
    log_min_shot_noise = torch.log10(torch.Tensor([0.0001]))
    #log_max_shot_noise = torch.log10(torch.Tensor([0.022]))
    log_max_shot_noise = torch.log10(torch.Tensor([0.020]))
    distribution = dist.uniform.Uniform(log_min_shot_noise, log_max_shot_noise)
    log_shot_noise = distribution.sample()
    shot_noise = torch.pow(10, log_shot_noise)
    distribution = dist.normal.Normal(torch.Tensor([0.0]), torch.Tensor([0.20]))
    read_noise = distribution.sample().clamp(min=-1.12, max=1.12)
    #line = lambda x: 1.85 * x + 0.30  # Line SIDD test set
    line = lambda x: 1.84 * x + 0.27  # Line SIDD test set
    log_read_noise = line(log_shot_noise) + read_noise
    read_noise = torch.pow(10, log_read_noise)
    return shot_noise, read_noise


def add_noise(image, shot_noise=0.01, read_noise=0.0005, use_cuda=False):
    """Adds random shot (proportional to image) and read (independent) noise."""
    variance = image * shot_noise + read_noise
    std = torch.sqrt(variance)
    mean = torch.Tensor([0.0])
    if use_cuda:
        mean = mean.cuda()
    distribution = dist.normal.Normal(mean, std)
    noise = distribution.sample()
    return image + noise, std


def add_rawnoise(image):
    """Adds random shot (proportional to image) and read (independent) noise."""
    if torch.rand(1) > 0.5:
        shot_noise, read_noise = random_noise_levels_sidd()
    else:
        shot_noise, read_noise = random_noise_levels_dnd()
    shot_noise = shot_noise.to(image.device)
    read_noise = read_noise.to(image.device)
    variance = image * shot_noise + read_noise
    std = torch.sqrt(variance)
    mean = torch.Tensor([0.0]).to(image.device)
    distribution = dist.normal.Normal(mean, std)
    noise = distribution.sample()
    return image + noise, noise, std




if __name__ == '__main__':
    from utils import utils_image as util
#    print(random_noise_levels_dnd())
    # run utils/utils_noise.py
    img = util.imread_uint('utils/test.bmp', 3)
#    img = util.imread_uint('utils/b.png', 3)
    img = util.uint2single(img)

    noise_level = 25

#    scale = power(10, noiseLevel);
#    noise = scale * imnoise(im2double(imgVec{j})/scale, 'poisson');

#    scale = 10**0.1
#    noise = scale *np.random.poisson(img/scale)

#    vals = len(np.unique(img))
#    vals = 2 ** np.ceil(np.log2(vals))
#    vals = 10**2  # [2, 4]
#    img = np.random.poisson(img * vals).astype(np.float32) / vals
#    img = util.single2uint(img)
#    util.imsave(img,'spec_noisy.png')
#    
    from scipy.linalg import orth
    L = 25/255
    D = np.diag(np.random.rand(3))
    U = orth(np.random.rand(3,3))
    conv = np.dot(np.dot(np.transpose(U), D), U)
#    conv = np.ones((3,3))
    imageNoiseSigma = np.abs(L**2*conv)
    img += np.random.multivariate_normal([0,0,0], imageNoiseSigma, img.shape[:2]).astype(np.float32)
    img = util.single2uint(img)
    util.imsave(img,'spec_noisy.png')
    
