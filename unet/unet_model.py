# # full assembly of the sub-parts to form the complete net
import torch as th
from unet.unet_parts import *
import torch.nn as nn
import numpy as np
import time

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
        self.outc = outconv(64, n_classes)
        self.loc = locconv(64)
        self.conf = confconv(64)
        # x_np = np.linspace(1, 128, 128)  # 解码过程
        # y_np = np.linspace(1, 128, 128)
        # cy_np, cx_np = np.meshgrid(x_np, y_np)
        # self.cx = th.cuda.FloatTensor(cx_np)
        # self.cy = th.cuda.FloatTensor(cy_np)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.resnet(x)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        mask = self.outc(x)
        loc, x = self.loc(x)
        conf = self.conf(x)
        # del_loc = del_box(self.cx, self.cy, loc, mask, 0.5, 0.2)
        return mask, loc, conf, x,

def del_box(cx, cy, loc, mask, mask_thresh, loc_thresh):
    t1 = time.time()
    loc = box_decoder(cx, cy, pre=loc, map_size=128, sub_size=16)
    t2 = time.time()
    loc = th.reshape(loc, (-1, 4))
    mask = th.reshape(mask, (-1, 1))
    t3 = time.time()
    mask_id = th.where(mask > th.ones_like(mask) * mask_thresh, th.ones_like(mask), th.zeros_like(mask)).byte()[:, 0]
    idx = th.squeeze(th.nonzero(mask_id))
    t4 = time.time()
    get_box = th.index_select(loc, 0, idx)
    # get_box_np = get_box.detach().cpu().numpy()
    t5 = time.time()
    print('decoder time:', t2 - t1, 'reshape time:', t3 - t2, 'mask time', t4 - t3, 'select time', t5 - t4,
          'total time:', t5 - t1)
    return get_box


def box_decoder(cx, cy, pre, map_size, sub_size):

    pre_cx = cx + pre[:, 0, ...] * map_size
    pre_cy = cy + pre[:, 1, ...] * map_size
    pre_w = pre[:, 2, ...] * sub_size
    pre_h = pre[:, 3, ...] * sub_size

    pre_xmin = (pre_cx - pre_w / 2)
    pre_xmax = (pre_cx + pre_w / 2)
    pre_ymin = (pre_cy - pre_h / 2)
    pre_ymax = (pre_cy + pre_h / 2)

    de_box = th.cat((pre_xmin, pre_xmax, pre_ymin, pre_ymax), dim=2)
    # t3 = time.time()
    # print('grid time:', t2 - t1, 'decoder time:', t3 - t2)
    return de_box

