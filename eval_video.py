import cv2
import numpy as np
from unet.unet_model3 import *
import time
from utils.NMS import *
import torch as th
import copy

path = 'E:\Person_detection\Dataset\\video\\test2.mp4'


unet = UNet(3, 1).to('cuda')
unet.eval()
unet.load_state_dict(th.load('.\checkpoint\\pretrain\\PersonMasker_model38.pt'))
conf = confconv(64).to('cuda')
conf.train()

cap = cv2.VideoCapture(path)
sum_time = 0
num_frame = 1
while(True):
     ret, frame = cap.read()                       # 一帧一帧读取视频
     frame_512 = np.expand_dims(np.transpose(cv2.resize(frame, (512, 512)), [2, 0, 1]), 0)
     t1 = time.time()
     mask_256, bbox_256, _, conf_256 = unet(th.cuda.FloatTensor(frame_512))
     conf_256 = conf(conf_256)
     t2 = time.time()
     box_512 = mask_nms(mask=mask_256, box=bbox_256, conf=conf_256, mask_thresh=0.5, conf_thresh=0.4, roi_thresh=0.2)
     t3  =time.time()
     mask = cv2.resize(np.transpose(mask_256.detach().cpu().numpy()[0, :, :, :], [1, 2, 0]), (512, 512))

     mask_per = np.repeat(np.expand_dims(np.where(mask > 0.5, 1, 0), -1), 3, -1).astype(np.uint8)
     mask_per2 = np.ones_like(mask_per) - mask_per
     mask_per[:, :, 0] = mask_per[:, :, 0] * 255
     mask_per[:, :, 1] = mask_per[:, :, 1] * 150
     # mask_per2 = np.ones_like(mask_per) - mask_per



     if num_frame >1:
          sum_time += (t3 - t1)
     print('current frame time:', (t2 - t1), 'NMS time:', (t3 - t2), 'avg frame time:', sum_time / num_frame)
     num_frame += 1
     mask_frame = np.transpose(frame_512[0, :, :, :], [1, 2, 0]) + mask_per

     cv2.imwrite('E:\Person_detection\Mask_Yolo\mask_image\\{}.jpg'.format(num_frame), mask_frame)

     # for box in box_512:
     #      cv2.rectangle(mask_frame, (box[2], box[0]), (box[3], box[1]), [255, 0, 0], 1)


     cv2.imshow('frame', mask_frame)                      # 显示结果
     if cv2.waitKey(1) & 0xFF == ord('q'):         # 按q停止
        break
cap.release()                                     #释放cap,销毁窗口
cv2.destroyAllWindows()
print()
