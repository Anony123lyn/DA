import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import datasets
from config import config
from config import update_config
from utils.utils import create_logger

from models.Base_VGG import DeepLabV2_VGG

from models.Source_Domain_DeepLabV2_ResNet import DeepLabV2_ResNet
from core.test_function_GTA_source import test


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    # prepare args & logs
    args = parse_args()
    logger, final_output_dir = create_logger(config, args.cfg, 'test')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark     = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled       = config.CUDNN.ENABLED

    # prepare data
    test_dataset = eval('datasets.' + config.DATASET.DATASET)(
                        root            = config.DATASET.ROOT,
                        anno_file       = config.DATASET.TEST_FILE,
                        is_train        = False,
                        image_dir       = config.DATASET.IMAGE_DIR,
                        anno_dir        = config.DATASET.ANNO_DIR,
                        crop_size       = tuple(config.DATASET.IMAGE_SIZE),
                        src_size       = tuple(config.DATASET.SRC_SIZE),
                        ignore_label    = config.TRAIN.IGNORE_LABEL,
                        num_samples     = config.TEST.NUM_SAMPLES,
                        test_scales     = config.TEST.TEST_SCALES,
                        test_flip       = config.TEST.TEST_FLIP)

    testloader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size      = 1,
                        shuffle         = False,
                        num_workers     = config.WORKERS,
                        pin_memory      = True)
       
   
    # init model
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = DeepLabV2_VGG().to(device=device)
    # model = DeepLabV2_ResNet().to(device=device)
    
    # model = DeepLabV2_ResNet().to(device=device)
    # model.create_architecture()
    # model = nn.DataParallel(model, device_ids=list(config.GPUS)).cuda()
    model = DeepLabV2_ResNet()
    model.create_architecture()
    model = nn.DataParallel(model, device_ids=list(config.GPUS)).cuda()

    # load whole model
    model_path = 'output/GTA/ResNet_city/best_51.91.pth'
    # model_path = 'output/GTA/LTIR_LT_adv/best_34.89.pth'

    # model_path = 'output/GTA/LTIR_Self/best111.pth'
    # model_path = 'output/GTA/LTIR_latent_only/best.pth'
    # model_path = 'output/GTA/LTIR_StoL/best_45.02.pth'
    # model_path = 'output/GTA/Vgg_realall_adv/best_26.47.pth'
    # model_path = 'output/GTA/LTIR_vgg_gta/best_22.48_27.87.pth'
    # model_path = 'output/GTA/LTIR_vgg_city/best_61.18.pth'
    # model_path = 'output/GTA/mul_adv_ocda/best.pth'
    # model_path = 'output/GTA/OCDA_MAML/best_0.47.pth'
    # model_path = 'output/GTA/Single_to_single_VGG/best_0.384_0.2.pth'
    # model_path = 'output/Dearmodel/new_0.372.pth'
####################################################################
    # model_path = 'output/GTA/LTIR_vgg_gta/best_22.48_27.87.pth'
    # model_path = 'output/GTA/OCDA_MAML/best_0.47.pth'
    # model_path = 'output/GTA/OCDA_self_multi_L/best_27.75.pth'

    pretrained_dict = torch.load(model_path)
    # dict_name = list(pretrained_dict)
    # for i, p in enumerate(dict_name):
    #     print(i, p)
    model_dict = model.state_dict()
    # dict_name = list(model_dict)
    # for i, p in enumerate(dict_name):
    #     print(i, p)
    logger.info('=> loading pretrained model {}'.format(model_path))
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    logger.info('=> loaded {} layers for model'.format(len(pretrained_dict.keys())))

    # model_dict = {}
    # state_dict = model.state_dict()
    # for k, v in pretrained_dict.items():
    #     for i, j in state_dict.items():   
    #         m = i[7:]
    #         print(i)
    #         print(m)
    #         print(k)
    #         print('1111111111111')
    #         if m == k :
    #             model_dict[i] = v
    #             # print(i)
    # state_dict.update(model_dict)
    # model.load_state_dict(state_dict)

    # logger.info('=> loaded {} layers for model'.format(len(state_dict.keys())))

    # evaluating
    start = timeit.default_timer()
    res_list = test(test_dataset, testloader, model)

    for idx, res in enumerate(res_list):
        msg = 'Output{} ==> MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, Mean_Acc: {: 4.4f}, Class IoU: '.format(str(idx), res[0], res[2], res[3])
        logging.info(msg)
        logging.info(res[1])

        # msg = 'Output{} ==> MeanIU_S0: {: 4.4f}, MeanIU_S1: {: 4.4f}, MeanIU_S2: {: 4.4f}, MeanIU_S3: {: 4.4f}, Class IoU_S0: , Class IoU_S1: , Class IoU_S2: , Class IoU_S3: '.format(str(idx), res[0], res[2], res[4], res[6])
        # logging.info(msg)
        # logging.info(res[1])
        # logging.info(res[3])
        # logging.info(res[5])
        # logging.info(res[7])

    end = timeit.default_timer()
    logger.info('Mins: %d' % np.int((end - start) / 60))
    logger.info('Done')

if __name__ == '__main__':
    main()
        