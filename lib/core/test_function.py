import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm
from config import config
import torch
import torch.nn as nn
from torch.nn import functional as F
import cv2
# from packaging import version
from core.inference import get_confusion_matrix
from utils.utils import AverageMeter, adjust_learning_rate
from utils.eval_utils import calc_miou_from_confusion_matrix
from utils.utils import get_affine_transform
from utils.modelsummary import get_model_summary
from sklearn.manifold import TSNE

from utils.misc import intersectionAndUnionGPU, get_color_pallete

def print_iou(iou, acc, miou, macc):
    for ind_class in range(iou.shape[0]):
        print('===> {0:2d} : {1:.2%} {2:.2%}'.format(ind_class, iou[ind_class, 0].item(), acc[ind_class, 0].item()))
    print('mIoU: {:.2%} mAcc : {:.2%} '.format(miou, macc))

def test(test_dataset, 
         testloader, 
         model):

    # evaluate
    model.eval()
    
    level_class = 19

    confusion_matrix = np.zeros((level_class, level_class))

    # if version.parse(torch.__version__) >= version.parse('0.4.0'):
    #     interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)
    # else:
    #     interp = nn.Upsample(size=(1024, 2048), mode='bilinear')
    

    with torch.no_grad():
        for i_iter, batch in enumerate(tqdm(testloader)):
            image, label, name = batch 

            size  = label.size()
            label = label.long().cuda()
            image = image.float().cuda()
            
            output = test_dataset.inference(model, image)
            # print(out_feature.shape)
    #         # # for parsing, image was resized to inference, and upsample to original size to eval
            if output.size()[-2] != size[-2] or output.size()[-1] != size[-1]:
                output = F.upsample(output, (size[-2], size[-1]), mode='bilinear')

            # # Evaluate human parsing
            confusion_matrix += get_confusion_matrix(label,
                                                     output,
                                                     size,
                                                     level_class,
                                                     config.TRAIN.IGNORE_LABEL)


            if i_iter % 100 == 0:
                logging.info('processing: %d images' % i_iter)
                miou, _, _, _ = calc_miou_from_confusion_matrix(confusion_matrix)
                logging.info('mIoU: {: 4.4f}'.format(miou))


    mean_IoU, IoU_array, pixel_acc, mean_acc = calc_miou_from_confusion_matrix(confusion_matrix)
    return [[mean_IoU, IoU_array, pixel_acc, mean_acc]]