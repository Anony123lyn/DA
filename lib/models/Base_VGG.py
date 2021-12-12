import math
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, inchannel, outchannel, num_classes):
        super(ASPP, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(inchannel, outchannel, kernel_size=1)
        self.ReLU_conv_1x1_1 = nn.ReLU(inplace=True)

        self.conv_3x3_1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=6, dilation=6)
        self.ReLU_conv_3x3_1 = nn.ReLU(inplace=True)

        self.conv_3x3_2 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=12, dilation=12)
        self.ReLU_conv_3x3_2 = nn.ReLU(inplace=True)

        self.conv_3x3_3 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=18, dilation=18)
        self.ReLU_conv_3x3_3 = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(inchannel, outchannel, kernel_size=1)
        self.ReLU_conv_1x1_2 = nn.ReLU(inplace=True)

        self.conv_1x1_3 = nn.Conv2d(outchannel*5, outchannel, kernel_size=1) # (1280 = 5*256)
        self.ReLU_conv_1x1_3 = nn.ReLU(inplace=True)

        self.conv_1x1_4 = nn.Conv2d(outchannel, num_classes, kernel_size=1)

    def forward(self, feature_map):
        # (feature_map has shape (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet instead is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8))

        feature_map_h = feature_map.size()[2] # (== h/16)
        feature_map_w = feature_map.size()[3] # (== w/16)

        out_1x1 = F.relu(self.ReLU_conv_1x1_1(self.conv_1x1_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = F.relu(self.ReLU_conv_3x3_1(self.conv_3x3_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = F.relu(self.ReLU_conv_3x3_2(self.conv_3x3_2(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(self.ReLU_conv_3x3_3(self.conv_3x3_3(feature_map))) # (shape: (batch_size, 256, h/16, w/16))

        out_img = self.avg_pool(feature_map) # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.ReLU_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1))
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, h/16, w/16))

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, 1280, h/16, w/16))
        out_feature = F.relu(self.ReLU_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, h/16, w/16))
        out = self.conv_1x1_4(out_feature) # (shape: (batch_size, num_classes, h/16, w/16))

        return out,out_feature



class DeepLabV2_VGG(nn.Module):

    def __init__(self):
        super(DeepLabV2_VGG, self).__init__()
        self.n_channels = 3
        self.n_classes = 19
        # self.bilinear = bilinear
        features = []
        features.append(nn.Conv2d(self.n_channels, 64, 3, stride=1, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(64, 64, 3, stride=1, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.MaxPool2d(2, stride=2, padding=0, dilation=1, ceil_mode=False))

        features.append(nn.Conv2d(64, 128, 3, stride=1, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(128, 128, 3, stride=1, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.MaxPool2d(2, stride=2, padding=0, dilation=1, ceil_mode=False))

        features.append(nn.Conv2d(128, 256, 3, stride=1, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(256, 256, 3, stride=1, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(256, 256, 3, stride=1, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.MaxPool2d(2, stride=2, padding=0, dilation=1, ceil_mode=False))

        features.append(nn.Conv2d(256, 512, 3, stride=1, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(512, 512, 3, stride=1, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(512, 512, 3, stride=1, padding=1))
        features.append(nn.ReLU(inplace=True))

        features.append(nn.Conv2d(512, 512, 3, stride=1, padding=2, dilation=2))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(512, 512, 3, stride=1, padding=2, dilation=2))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(512, 512, 3, stride=1, padding=2, dilation=2))
        features.append(nn.ReLU(inplace=True))

        features.append(nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4))
        features.append(nn.ReLU(inplace=True))        
        self.features = nn.Sequential(*features)

        self.aspp = ASPP(1024,1024,self.n_classes)
        

    def forward(self, x):
        _, _, h, w = x.size()
        out = self.features(x)
        out,out_feature = self.aspp(out)
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        return out

    # def _init_weights(self):
    #     def normal_init(module):
    #         for m in module:
    #             if isinstance(m, nn.Conv2d):
    #                 nn.init.normal_(m.weight, std=0.001)
    #             elif isinstance(m, nn.BatchNorm2d):
    #                 nn.init.constant_(m.weight, 1)
    #                 nn.init.constant_(m.bias, 0)

    # def create_architecture(self):
    #     self._init_modules()
    #     self._init_weights()
    
    # if __name__ == "__main__":

    # DeepLabV2_VGG = DeepLabV2_VGG()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")