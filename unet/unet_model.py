# # full assembly of the sub-parts to form the complete net
import torch as th
from unet.unet_parts import *
import torch.nn as nn



resnet = resnet18()
resnet.load_state_dict(th.load('E:\Person_detection\Pytorch-UNet\\checkpoint\\pretrain\\resnet18-5c106cde.pth'))
resnet.eval()



class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        self.resnet = resnet
        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)
        # self.up4 = up(64, 64)
        self.outc = outconv(64, n_classes)
        self.loc = locconv(64)
        self.conf = confconv(64)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.resnet(x)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        # x = self.up4(x, x1)
        cls = self.outc(x)
        loc = self.loc(x)
        conf = self.conf(x)
        return cls, loc, conf
