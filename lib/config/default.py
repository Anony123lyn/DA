
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.GPUS = (0,)
_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.VIS_DIR = ''

_C.WORKERS = 4
_C.PRINT_FREQ = 100
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.NET = ''
# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# DATASET related params
_C.DATASET = CN()
_C.DATASET.DATASET = 'GTA'
_C.DATASET.ROOT = ''
_C.DATASET.NUM_CLASSES = 20

_C.DATASET.TRAIN_FILE = ''
_C.DATASET.TEST_FILE = ''
_C.DATASET.SOURCE_FILE = ''
_C.DATASET.TARGET_FILE = ''

_C.DATASET.TARGET_FILE_aug = ''
_C.DATASET.TARGET_FILE_rainy = ''
_C.DATASET.TARGET_FILE_snowy = ''
_C.DATASET.TARGET_FILE_cloudy = ''
_C.DATASET.TARGET_FILE_overcast = ''

_C.DATASET.WEAK_DIR = ''
_C.DATASET.ANNO_DIR = ''
_C.DATASET.IMAGE_DIR = ''
_C.DATASET.WEAK_FILE = ''


_C.DATASET.BASE_SIZE = 600
_C.DATASET.IMAGE_SIZE = [600, 600]
_C.DATASET.SRC_SIZE = [600, 600]
_C.DATASET.HEATMAP_SIZE = [600, 600]
_C.DATASET.SIGMA = 5
_C.DATASET.POSE_SCALE_FACTOR = 1.0
_C.DATASET.PARS_SCALE_FACTOR = 11
_C.DATASET.ROTATION_FACTOR = 1
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROTATION_FACTOR = 30

_C.LOSS = CN()
_C.LOSS.USE_OHEM = True
_C.LOSS.OHEMTHRES = 0.9
_C.LOSS.OHEMKEEP = 131072

# common params for NETWORK
_C.PARS_MODEL = CN()
_C.PARS_MODEL.NAME = 'dtcf'
_C.PARS_MODEL.PRETRAINED = ''
_C.PARS_MODEL.EXTRA = CN(new_allowed=True)

_C.TRAIN = CN()
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.01
_C.TRAIN.LR_inner = 0.0001
_C.TRAIN.LR_outer = 0.00005
_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.IGNORE_LABEL = 255
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.END_EPOCH = 20
_C.TRAIN.RESUME = False
_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.SHUFFLE = True
_C.TRAIN.NUM_SAMPLES = 0

# testing
_C.TEST = CN()
_C.TEST.IGNORE_LABEL = -1
_C.TEST.BATCH_SIZE = 1
_C.TEST.NUM_SAMPLES = 0
_C.TEST.MODEL_FILE = ''
_C.TEST.TEST_SCALES = [1]
_C.TEST.TEST_FLIP = False
_C.TEST.TEST_NET = ''

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


def update_config(cfg, args):
    cfg.defrost()
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

