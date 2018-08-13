import os
import cv2
import scipy.io as sio
import numpy as np
Bbox_path = 'E:\Person_detection\Dataset\DataSets2017\\u_net\sub_bbox_64'
Image_path = 'E:\Person_detection\Dataset\DataSets2017\\u_net\sub_image_64'
test = 'E:\Person_detection\Dataset\DataSets2017\\u_net\\test'

bbox_list = os.listdir(Bbox_path)
image_list = os.listdir(Image_path)
idx = 0
for name_b, name_i in zip(bbox_list, image_list):
    print(idx)

    bbox = sio.loadmat(os.path.join(Bbox_path, name_b))['bbox']
    image = cv2.resize(cv2.imread(os.path.join(Image_path, name_i)), (32, 32))
    for i in range(bbox.shape[0]):
        for j in range(bbox.shape[1]):
            if np.abs(bbox[i, j, 0]) > 0:
                x = int(i + bbox[i, j, 0] * 256)
                y = int(j + bbox[i, j, 1] * 256)
                w = (bbox[i, j, 2] * 32)
                h = (bbox[i, j, 3] * 32)

                ymin = min(int(y - h / 2), 31)
                xmin = min(int(x - w / 2), 31)
                xmax = min(int(x + w / 2), 31)
                ymax = min(int(y + h / 2), 31)
                # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), [255, 0, 0], 1)


                image[xmin:xmax, ymin, :] = [0, 0, 0]
                image[xmin:xmax, ymax, :] = [0, 0, 0]
                image[xmin, ymin:ymax, :] = [0, 0, 0]
                image[xmax, ymin:ymax, :] = [0, 0, 0]
    save_path = os.path.join(test, str(idx) + '.jpg')
    cv2.imwrite(save_path, image)
    idx += 1




