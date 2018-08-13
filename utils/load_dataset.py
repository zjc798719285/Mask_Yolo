import cv2, os
import numpy as np
import math


def load_dataset(ImagePath, MaskPath):
    imgs = os.listdir(ImagePath)
    masks = os.listdir(MaskPath)
    dataset = []
    for idx, (img_i, mask_i) in enumerate(zip(imgs, masks)):
        print(idx)
        image = np.transpose(cv2.imread(os.path.join(ImagePath, img_i)), [2, 0, 1])
        mask = np.transpose(cv2.resize(cv2.imread(os.path.join(MaskPath, mask_i))/255, (image.shape[1]//2, image.shape[2]//2)), [2, 0, 1])
        # mask = cv2.imread(os.path.join(MaskPath, mask_i))[:, :, 0] / 255
        mask = np.where(mask > 0.5, 1, 0)
        dataset.append([image, mask])
    return dataset



def split_train_val(dataset, val_percent=0.01):
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

    def next_batch(self):
        if self.step > int(math.floor(len(self.dataset) / self.batch_size)):
            self.shuffle_data()
        start = (self.step - 1) * self.batch_size
        stop = self.step * self.batch_size
        batch = self.dataset[start:stop]
        image = []; mask = []
        for batch_i in batch:
            image.append(batch_i[0])
            mask.append(batch_i[1])
        image = np.array(image)
        mask = np.array(mask)
        self.step += 1
        return image, mask



    def next_batch_cat(self, scale, im_size):

        if self.step > int(math.floor(len(self.dataset) / self.batch_size)):
            self.shuffle_data()
        start = (self.step - 1) * self.batch_size
        stop = self.step * self.batch_size
        batch = self.dataset[start:stop]
        image = []; mask = []
        batch_size = int(self.batch_size / scale / scale)
        idx = 0;cat_img = np.zeros(shape=(3, im_size, im_size))
        cat_mask = np.zeros(shape=(3, im_size//2, im_size//2))
        crop_size_img = int(im_size / scale)
        crop_size_mask = int(im_size / scale/2)
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
                    idx += 1
            image.append(cat_img)
            mask.append(cat_mask)
        image = np.array(image)
        mask = np.array(mask)
        self.step += 1
        return image, mask









if __name__ =='__main__':

    ImagePath = 'E:\Person_detection\Dataset\DataSets2017\\u_net\sub_image'
    MaskPath = 'E:\Person_detection\Dataset\DataSets2017\\u_net\sub_mask'
    dataset = load_dataset(ImagePath, MaskPath)
    train_set, val_set = split_train_val(dataset, val_percent=0.05)
    trainLoader = DataLoader(train_set, 64)
    for _ in range(10):
        image, mask = trainLoader.next_batch_cat(8, 512)
        img = np.transpose(image[0, :, :, :], [1, 2, 0])
        mask = np.transpose(mask[0, :, :, :], [1, 2, 0]) * 255
        # res_image = np.transpose(np.reshape(image, (1, 3, 512, 512))[0, ...], [1, 2, 0])
        cv2.imwrite('E:\Person_detection\Pytorch-UNet\\test_image1.jpg', img)
        cv2.imwrite('E:\Person_detection\Pytorch-UNet\\mask_image1.jpg', mask)
        print()
