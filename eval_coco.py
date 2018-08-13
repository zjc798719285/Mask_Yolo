import torch as th
import os
import cv2
import numpy as np
from unet.unet_model import UNet
import time

unet = UNet(3, 1).to('cuda')
unet.eval()
unet.load_state_dict(th.load('.\checkpoint\\PersonMasker262.pt'))


evalImagePath = 'E:\Person_detection\Dataset\DataSets2017\\u_net\image'
evalMaskPath = 'E:\Person_detection\Pytorch-UNet\eval\mask_coco'
imgs = [os.path.join(evalImagePath, i) for i in os.listdir(evalImagePath)]
for idx, img_i in enumerate(imgs):
    img = np.expand_dims(np.transpose(cv2.imread(img_i), [2, 0, 1]), 0)
    t1 = time.time()
    mask = unet(th.cuda.FloatTensor(img))
    t2 = time.time()
    mask = cv2.resize(np.transpose(np.repeat(mask.detach().cpu().numpy()[0, :, :, :], 3, 0), [1, 2, 0]), (412, 412))
    background = np.zeros_like(mask)
    color = np.ones_like(mask);color[:, :, 0] = 150;color[:, :, 1] = 50;color[:, :, 2] = 170
    mask = np.where(mask > 0.5, color, background)
    img = np.transpose(img[0, :, :, :], [1, 2, 0])
    mask_img = mask + img
    cv2.imwrite(os.path.join(evalMaskPath, '{}.jpg'.format(idx)), mask_img)
    print('id', idx, 'time:', t2 - t1)



