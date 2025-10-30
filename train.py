import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader

import argparse
import logging
import warnings

from data import dataloader_pairs 
from utils import utils_logger
from utils.utils import setup_seed
from model.model_dynamic import ModelPlain as M

from utils import utils_image as util
from utils import utils_option as option



def main(config):
    # ----------------------------------------
    # seed
    # ----------------------------------------
    setup_seed(2022)
    # ----------------------------------------
    # create dataset
    # ----------------------------------------
    train_dataset = dataloader_pairs.dataloader(config.lowlight_images_path, config.overlight_images_path, config.normallight_images_path, config.flow_path, patch_size = config.patch_size, phase='train')   
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    test_dataset = dataloader_pairs.dataloader(config.test_lowlight_images_path, config.test_overlight_images_path, config.test_normallight_images_path, flow_path=None, patch_size = config.patch_size, phase='val')   
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    # ----------------------------------------
    # initialize model
    # ----------------------------------------
    init_epoch_A, init_path_A = option.find_last_checkpoint(config.model_save_dir, net_type='A')
    init_epoch_F, init_path_F = option.find_last_checkpoint(config.model_save_dir, net_type='F')
    init_epoch_D, init_path_D = option.find_last_checkpoint(config.model_save_dir, net_type='D')
    init_epoch_G, init_path_G = option.find_last_checkpoint(config.model_save_dir, net_type='G')

    if config.train_stage == 'align':
        current_epoch = max(init_epoch_A, init_epoch_F)
    elif config.train_stage == 'fusion':
        current_epoch = max(init_epoch_D, init_epoch_G)
    elif config.train_stage == 'joint':
        current_epoch = max(init_epoch_A, init_epoch_F, init_epoch_D, init_epoch_G)
    else:
        raise NotImplementedError('config.train_stage [{:s}] is not found.'.format(config.train_stage))

    model_plain = M(config)
    model_plain.init_train(init_paths=[init_path_A, init_path_F, init_path_D, init_path_G])
    model_plain.print_network()

    # ----------------------------------------
    # training
    # ----------------------------------------
    for epoch in range(current_epoch+1, config.num_epochs+1):
        # -------------------------------
        # 1) update learning rate
        # -------------------------------
        model_plain.update_learning_rate(epoch)
        
        for iteration, train_data in enumerate(train_loader): 
            
            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model_plain.feed_data(train_data)
            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model_plain.optimize_parameters()
            # -------------------------------
            # 4) training information
            # -------------------------------
            logs = model_plain.current_log()  
            message = '<epoch:{:3d}, iter:{:3,d}, lr:{:.3e}> '.format(epoch, iteration, model_plain.current_learning_rate())
            for k, v in logs.items():  
                message += '{:s}: {:.3e} '.format(k, v)
            logger.info(message)


        if (epoch % config.display_epoch) == 0:
            # -------------------------------
            # 5) save model
            # -------------------------------
            model_plain.save(epoch=epoch)
            # -------------------------------
            # 6) testing
            # -------------------------------
            avg_psnr = 0.0
            avg_ssim = 0.0
            idx = 0
            for test_data in test_loader:
                idx += 1
                under, over, gt, image_path = test_data
                image_name_ext = os.path.basename(image_path[0])
                img_name, ext = os.path.splitext(image_name_ext)
                img_dir = os.path.join(config.image_save_dir, img_name)
                util.mkdir(img_dir)
                
                model_plain.feed_data(test_data)
                model_plain.test()

                visuals = model_plain.current_visuals()
                E_img = util.tensor2uint(visuals['E'])
                H_img = util.tensor2uint(visuals['H'])
                # -----------------------
                # save estimated image E
                # -----------------------
                save_img_path = os.path.join(img_dir, '{:s}_{:03d}.png'.format(img_name, epoch))
                util.imsave(E_img, save_img_path)
                
                # -----------------------
                # calculate PSNR
                # -----------------------
                current_psnr = util.calculate_psnr(E_img, H_img)
                current_ssim = util.calculate_ssim(E_img, H_img)
                
                logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB | {:<4.3f}'.format(idx, image_name_ext, current_psnr, current_ssim))
                
                avg_psnr += current_psnr
                avg_ssim += current_ssim
                
            avg_psnr = avg_psnr / idx
            avg_ssim = avg_ssim / idx
            
            logger.info('<epoch:{:3d}, Average PSNR : {:<.2f}dB, Average SSIM : {:<.3f}\n'.format(epoch, avg_psnr, avg_ssim))
        
                


if __name__ == "__main__": 
    warnings.filterwarnings("ignore")
    task_name="train1"

    parser = argparse.ArgumentParser(description='PyTorch implementation of TextMEF')

    parser.add_argument('--task_name', type=str, default="train0") 
    parser.add_argument('--train_stage', type=str, default='joint', choices=['fusion', 'align', 'joint']) 
    parser.add_argument('--lowlight_images_path', type=str, default="Dataset/train_data/SICE/trainA/") 
    parser.add_argument('--overlight_images_path', type=str, default="Dataset/train_data/SICE/trainB/")
    parser.add_argument('--normallight_images_path', type=str, default="Dataset/train_data/SICE/trainC/") 
    parser.add_argument('--flow_path', type=str, default="Dataset/train_data/SICE/flow/")
    parser.add_argument('--test_lowlight_images_path', type=str, default="Dataset/test_data/real/under/") 
    parser.add_argument('--test_overlight_images_path', type=str, default="Dataset/test_data/real/over/") 
    parser.add_argument('--test_normallight_images_path', type=str, default="Dataset/test_data/real/gt/")
    parser.add_argument('--length_prompt', type=int, default=16)
    parser.add_argument('--train_lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--display_epoch', type=int, default=20)
    parser.add_argument('--image_save_dir', type=str, default="./"+task_name+"/"+"images/")
    parser.add_argument('--model_save_dir', type=str, default="./"+task_name+"/"+"models/")
    config = parser.parse_args()

    os.makedirs(config.image_save_dir, exist_ok=True)
    os.makedirs(config.model_save_dir, exist_ok=True)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(task_name, logger_name+'.log'))
    logger = logging.getLogger(logger_name)


    main(config)
