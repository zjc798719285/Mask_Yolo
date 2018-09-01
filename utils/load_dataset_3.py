import cv2, os
import numpy as np
import math
import scipy.io as sio
import copy
import gc
import time

def load_dataset(ImagePath, MaskPath, BboxPath, s_scale=4):
    '''
    :param ImagePath:
    :param MaskPath:
    :param BboxPath:
    :param s_scale: 子图像的缩放比,在我们这个网络上缩放比为4
    :return:
    '''
    imgs = os.listdir(ImagePath)
    masks = os.listdir(MaskPath)
    bbox = os.listdir(BboxPath)
    dataset = []
    for idx, (img_i, mask_i, bbox_i) in enumerate(zip(imgs, masks, bbox)):
        gc.disable()
        if idx == 25000:
            break
        print(idx)
        image = np.transpose(cv2.imread(os.path.join(ImagePath, img_i)), [2, 0, 1])
        mask = np.transpose(cv2.resize(cv2.imread(os.path.join(MaskPath, mask_i))/255, (image.shape[1]//s_scale, image.shape[2]//s_scale)), [2, 0, 1])
        bbox = np.transpose(sio.loadmat(os.path.join(BboxPath, bbox_i))['bbox'], [2, 0, 1])
        mask = np.where(mask > 0.5, 1, 0)
        dataset.append([image, mask, bbox])
    gc.enable()
    return dataset


#
# def load_dataset(ImagePath, MaskPath, BboxPath):
#     imgs = os.listdir(ImagePath)
#     masks = os.listdir(MaskPath)
#     bbox = os.listdir(BboxPath)
#     dataset = []
#     for idx, (img_i, mask_i, bbox_i) in enumerate(zip(imgs, masks, bbox)):
#         image = os.path.join(ImagePath, img_i)
#         mask = os.path.join(MaskPath, mask_i)
#         bbox = os.path.join(BboxPath, bbox_i)
#         dataset.append([image, mask, bbox])
#
#     return dataset





def split_train_val(dataset, val_percent):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    # np.random.shuffle(dataset)
    return dataset[:-n], dataset[-n:]




class DataLoader(object):

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.step = 1
        self.num_step = int(np.ceil(len(self.dataset) / batch_size))
        self.shuffle_data()


    def shuffle_data(self):
        np.random.shuffle(self.dataset)
        self.step = 1


    def get_batch(self, datalist, s_scale):
        batch = []

        for data_i in datalist:
            gc.disable()
            # t1 = time.time()
            image = np.transpose(cv2.imread(os.path.join(data_i[0])), [2, 0, 1])
            # t2 = time.time()
            # print('get_batch:', t2 - t1)
            mask = np.transpose(cv2.resize(cv2.imread(os.path.join(data_i[1]))/255,
                                          (image.shape[1]//s_scale, image.shape[2]//s_scale)), [2, 0, 1])
            mask = np.where(mask > 0.5, 1, 0)
            bbox = np.transpose(sio.loadmat(os.path.join(data_i[2]))['bbox'], [2, 0, 1])
            batch.append([image, mask, bbox])
        gc.enable()
        return batch

    def next_batch_cat(self, scale, im_size, scale_out):
        '''
        :param scale: 子图像和大图像比例，子图像为64*64，大图为512*512， scale=8
        :param im_size: 大图像的尺寸，本项目设定为512*512
        :param scale_out: Unet输入与输出比例，大图像为512*512，输出为128*128，scale_out=4
        :return:
        '''

        if self.step > int(math.floor(len(self.dataset) / self.batch_size)):
            self.shuffle_data()
        start = (self.step - 1) * self.batch_size
        stop = self.step * self.batch_size
        batch = self.dataset[start:stop]
        image = []; mask = []; bbox = []
        batch_size = int(self.batch_size / scale / scale)
        idx = 0;cat_img = np.zeros(shape=(3, im_size, im_size))
        cat_mask = np.zeros(shape=(3, im_size // scale_out, im_size // scale_out))
        cat_bbox = np.zeros(shape=(4, im_size // scale_out, im_size // scale_out))
        crop_size_img = int(im_size / scale)
        crop_size_mask = int(im_size / scale/scale_out)
        # crop_size_bbox = int(im_size / scale / 2)
        for i in range(batch_size):
            for j in range(scale):
                for k in range(scale):
                    xmin_img = j * crop_size_img
                    xmax_img = (j + 1) * crop_size_img
                    ymin_img = k * crop_size_img
                    ymax_img = (k + 1) * crop_size_img

                    xmin_mask = j * crop_size_mask
                    xmax_mask = (j + 1) * crop_size_mask
                    ymin_mask = k * crop_size_mask
                    ymax_mask = (k + 1) * crop_size_mask

                    cat_img[:, xmin_img:xmax_img, ymin_img:ymax_img] = batch[idx][0]
                    cat_mask[:, xmin_mask:xmax_mask, ymin_mask:ymax_mask] = batch[idx][1]
                    cat_bbox[:, xmin_mask:xmax_mask, ymin_mask:ymax_mask] = batch[idx][2]
                    idx += 1
            image.append(copy.deepcopy(cat_img))
            mask.append(copy.deepcopy(cat_mask))
            bbox.append(copy.deepcopy(cat_bbox))
        image = np.array(image)
        mask = np.array(mask)
        bbox = np.array(bbox)
        self.step += 1
        return image, mask, bbox









if __name__ =='__main__':

    ImagePath = 'E:\Person_detection\Dataset\DataSets2017\\u_net\sub_image_128'
    MaskPath = 'E:\Person_detection\Dataset\DataSets2017\\u_net\sub_mask_128'
    BboxPath = 'E:\Person_detection\Dataset\DataSets2017\\u_net\\sub_bbox_128_U'
    dataset = load_dataset(ImagePath, MaskPath, BboxPath)
    train_set, val_set = split_train_val(dataset, val_percent=0.05)
    trainLoader = DataLoader(train_set, 16*1)
    for _ in range(10):
        image, mask, bbox = trainLoader.next_batch_cat(4, 512, 4)
        img = np.transpose(image[0, :, :, :], [1, 2, 0])
        mask = np.transpose(mask[0, :, :, :], [1, 2, 0]) * 255
        # res_image = np.transpose(np.reshape(image, (1, 3, 512, 512))[0, ...], [1, 2, 0])
        for i in range(bbox.shape[2]):
            for j in range(bbox.shape[3]):
                if np.abs(bbox[0, 0, i, j]) > 0:
                    xmin = min(int((i - bbox[0, 0, i, j] * 128)), 127)
                    xmax = min(int((i + bbox[0, 1, i, j] * 128)), 127)
                    ymin = min(int((j - bbox[0, 2, i, j] * 128)), 127)
                    ymax = min(int((j + bbox[0, 3, i, j] * 128)), 127)

                    # ymin = min(int(y-h/2), 127)
                    # xmin = min(int(x-w/2), 127)
                    # xmax = min(int(x+w/2), 127)
                    # ymax = min(int(y+h/2), 127)

                    # mask[xmin:xmax, ymin:ymax, :] = [0, 0, 0]
                    # mask[x:x+5                    , y: y + 5,:] = [0, 0, 0]

                    mask[xmin:xmax, ymin, :] = [0, 0, 0]
                    mask[xmin:xmax, ymax, :] = [0, 0, 0]
                    mask[xmin, ymin:ymax, :] = [0, 0, 0]
                    mask[xmax, ymin:ymax, :] = [0, 0, 0]



        cv2.imwrite('E:\Person_detection\Mask_Yolo\\test_image1.jpg', img)
        cv2.imwrite('E:\Person_detection\Mask_Yolo\\mask_image1.jpg', mask)
        print()
