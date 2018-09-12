# # full assembly of the sub-parts to form the complete net

from unet.unet_parts4 import *
import torch.nn as nn

#增加通道Attention机制

resnet = resnet34()
resnet.load_state_dict(th.load('E:\Person_detection\Mask_Yolo\\checkpoint\\pretrain\\resnet34.pth'))
resnet.eval()



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, tensor_layer):
        super(UNet, self).__init__()
        self.resnet = resnet
        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)
        self.maskConvLSTM = maskConvLSTM(64, n_classes)
        self.maskConv = maskConv(128, n_classes)
        self.mask_h = tensor_layer[:, 0:1, :, :].to('cuda')
        self.mask_c = tensor_layer[:, 1:2, :, :].to('cuda')
        self.loc_h = tensor_layer[:, 2:6, :, :].to('cuda')
        self.loc_c = tensor_layer[:, 6:10, :, :].to('cuda')
        self.locConvLSTM = locConvLSTM(64, 4)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.resnet(x)
        up1 = self.up1(x5, x4)
        up2 = self.up2(up1, x3)
        up3 = self.up3(up2, x2)
        mask_h, mask_c = self.maskConvLSTM(up3, self.mask_h, self.mask_c)
        loc_h, loc_c = self.locConvLSTM(up3, self.loc_h, self.loc_c)
        self.mask_h = mask_h.detach()
        self.mask_c = mask_c.detach()
        self.loc_h = loc_h.detach()
        self.loc_c = loc_c.detach()
        mask_ = self.maskConv(up2)          #中间层mask输出

        return mask_h, loc_h, mask_

