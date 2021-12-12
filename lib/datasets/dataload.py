import torch.utils.data as data
import torch
import torch.utils.data
import scipy.io as io
from torch.nn import functional as F
from config import config
from PIL import Image
import cv2
import numpy as np
import os
import os.path
import json
import random
import time

# Use PIL to load image
def pil_loader(path):
    return Image.open(path).convert('RGB')

# Use opencv to load image
def opencv_loader(path):
    return cv2.imread(path, 1)

# LIP dataset Pose and Parsing
class GTA(data.Dataset):
    def __init__(self, root, anno_file, is_train=True,
                                        image_dir='.',
                                        anno_dir='.',
                                        mean=[104.00698793, 116.66876762, 122.67891434], 
                                        std=[0.225, 0.224, 0.229],
                                        scale_factor=11,
                                        loader=opencv_loader,
                                        crop_size='.',
                                        src_size='.',
                                        ignore_label=255,
                                        num_samples=0,
                                        test_scales=[1],
                                        test_flip=True):

        # Load train json file
        print('Loading {0} json file: {1}...'.format('training' if is_train else 'testing', anno_file))
        train_list = [line.strip().split() for line in open(anno_file)]
        print('Finished loading {0} json file, {1} images loaded, so good luck :)'.format('training' if is_train else 'testing', len(train_list)))

        # Origin-parameters
        self.num_classes        = 19
        self.name_classes    = ['0', '1', '2', '3', '4',
                                '5', '6', '7', '8', '9',
                                '10', '11', '12', '13', '14',
                                '15', '16', '17', '18']

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        

        # Hyper-parameters
        self.root               = root
        self.anno_list          = train_list
        self.is_train           = is_train
        self.image_dir          = image_dir
        self.anno_dir           = anno_dir
        self.mean               = mean
        self.std                = std
        self.scale_factor       = scale_factor
        self.loader             = loader
        self.ignore_label       = ignore_label
        self.crop_size          = crop_size
        self.src_size           = src_size
        self.num_samples        = num_samples
        self.test_scales        = test_scales
        self.test_flip          = test_flip

        if self.num_samples > 0:
            self.anno_list = self.anno_list[:self.num_samples]

        # Number of train samples
        self.N_train = len(self.anno_list)

    def __len__(self):
        return self.N_train
    
    def __getitem__(self, index):
        # Select a training sample
        name = self.anno_list[index][0]
        
        # ###########   Cityscapes val set ###################
        # pars_im   = self.loader(os.path.join(self.root, self.image_dir, 'Cityscapes/IMAGE', name + '_leftImg8bit.png'))
        # pars_an   = cv2.imread( os.path.join(self.root, self.anno_dir,  'Cityscapes/Label_gtfine', name + '_gtFine_labelIds.png'), 0)
        # pars_an = self.convert_label(pars_an)
        # pars_im   = cv2.resize(pars_im, self.src_size, interpolation = cv2.INTER_LINEAR)
        # ####################################################

        ###########  C-Driving test set #####################
        pars_im   = self.loader(os.path.join(self.root, self.image_dir, 'Real/IMAGE', name + '.jpg'))
        pars_an   = cv2.imread( os.path.join(self.root, self.anno_dir,  'Real/LABEL', name + '.png'), 0)
        pars_im   = cv2.resize(pars_im, self.src_size, interpolation = cv2.INTER_LINEAR)
        ####################################################
                               
        pars_im = self.input_transform(pars_im)
        pars_im = pars_im.transpose((2, 0, 1))
        
        return pars_im.copy(), pars_an.copy(), name        
    
    def convert_label(self, label):
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
             label_copy[label == k] = v
        return  label_copy

    def input_transform(self, image):
        # for opencv, trans BGR to RGB
        image = image.astype(np.float32)[:, :, ::-1]

        return image



    def inference(self, model, image):
        size = image.size()
        image = image.cuda()

        pars_pred = model(image)

        # inferece with fliping
        if self.test_flip:
            flip_image = torch.from_numpy(image.cpu().numpy()[:,:,:,::-1].copy()).cuda()
            flip_pars_output  = model(flip_image)          
            flip_pars_pred = flip_pars_output.cpu().numpy()[:,:,:,::-1]
            flip_pars_pred = torch.from_numpy(flip_pars_pred.copy()).cuda()

            pars_pred += flip_pars_pred
            pars_pred = pars_pred * 0.5
            
        return pars_pred.exp()
