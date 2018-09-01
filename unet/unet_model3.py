# # full assembly of the sub-parts to form the complete net
import torch as th
from unet.unet_parts3 import *
import torch.nn as nn
import numpy as np
import time
#增加通道Attention机制
resnet = resnet18()
resnet.load_state_dict(th.load('E:\Person_detection\Mask_Yolo\\checkpoint\\pretrain\\resnet18.pth'))
resnet.eval()



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.resnet = resnet
        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)
        self.outc = maskConv(64, n_classes)
        self.mask_h = th.zeros(6, 1, 128, 128).to('cuda')
        self.mask_c = th.zeros(6, 1, 128, 128).to('cuda')
        self.loc = locConv(64, 4)
        self.loc_h = th.zeros(6, 4, 128, 128).to('cuda')
        self.loc_c = th.zeros(6, 4, 128, 128).to('cuda')


    def forward(self, x):
        x1, x2, x3, x4, x5 = self.resnet(x)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        mask_h, mask_c = self.outc(x, self.mask_h, self.mask_c)
        loc_h, loc_c = self.loc(x, self.loc_h, self.loc_c)
        self.mask_h = mask_h.detach()
        self.mask_c = mask_c.detach()
        self.loc_h = loc_h.detach()
        self.loc_c = loc_c.detach()
        # self.mask_h.requires_grad = False
        # self.mask_c.requires_grad = False
        # self.loc_h.requires_grad = False
        # self.loc_c.requires_grad = False
        return mask_h, loc_h

