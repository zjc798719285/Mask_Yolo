import cv2
from unet.unet_model3 import *
import time
from utils.NMS import mask_nms as mask_nms
import torch as th
import numpy as np
import os

img_dir = 'G:\DataSet\coco\DataSets2017\JPEGImages\\'
output_dir = ''

tensor = th.zeros(1, 10, 128, 128)
unet = UNet(3, 1, tensor).to('cuda')
unet.eval()
unet.load_state_dict(th.load('E:\Person_detection\Mask_Yolo\checkpoint\model3\\PersonMaskerUnitBox_223.pt'))

imgs = [os.path.join(img_dir, i) for i in os.listdir(img_dir)]
                        # 一帧一帧读取视频
for img_i in imgs:
     img = cv2.imread(img_i)
     img_512 = np.repeat(np.expand_dims(np.transpose(cv2.resize(img, (512, 512)), [2, 0, 1]), 0), 1, 0)
     t1 = time.time()
     mask_128, bbox_128 = unet(th.cuda.FloatTensor(img_512))
     bbox_128_np = bbox_128.detach().cpu().numpy()
     norm_vec = (bbox_128_np[:, 1, ...] - bbox_128_np[:, 0, ...])**2 + \
                (bbox_128_np[:, 3, ...] - bbox_128_np[:, 2, ...])**2


     t2 = time.time()
     box_frame = mask_nms(mask=mask_128, box=bbox_128, mask_thresh=0.5,
                          iou_thresh=0.4, e_thresh=1.2, duty_thresh=0.1,
                          frame_shape=(img.shape[0], img.shape[1]))
     mask = cv2.resize(np.transpose(mask_128.detach().cpu().numpy()[0, :, :, :], [1, 2, 0]), (img.shape[1], img.shape[0]))



     t3 = time.time()
     mask_per = np.repeat(np.expand_dims(np.where(mask > 0.5, 1, 0), -1), 3, -1).astype(np.uint8)
     mask_per2 = np.ones_like(mask_per) - mask_per
     mask_per[:, :, 0] = mask_per[:, :, 0] * 20
     mask_per[:, :, 2] = mask_per[:, :, 2] * 0
     mask_per[:, :, 1] = mask_per[:, :, 1] * 50
     # mask_per2 = np.ones_like(mask_per) - mask_per

     # if num_frame > 1:
     #      sum_time += (t3 - t1)
     # print('current frame time:', (t2 - t1), 'NMS time:', (t3 - t2), 'avg frame time:', sum_time / num_frame)
     # num_frame += 1
     mask_frame = cv2.resize(img, (img.shape[1], img.shape[0])) + mask_per

     # cv2.imwrite('E:\Person_detection\Mask_Yolo\mask_image\\{}.jpg'.format(num_frame), mask_frame)

     for box in box_frame:
        cv2.rectangle(mask_frame, (box[2], box[0]), (box[3], box[1]), [255, 0, 0], 1)


     cv2.imshow('frame', mask_frame)                      # 显示结果
     if cv2.waitKey(1) & 0xFF == ord(' '):         # 按q停止
        while True:
             if cv2.waitKey(1) & 0xFF == ord(' '):
                  break
# cap.release()                                     #释放cap,销毁窗口
cv2.destroyAllWindows()
print()
