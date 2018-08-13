import numpy as np



def recall_ap(pre, target, cls):
    pre = pre[:, cls, :, :]
    target = target[:, cls, :, :]
    pre_one = np.where(pre > 0.5, 1, 0)                      #预测大于0.5，置1
    target_one = np.where(target > 0.5, 1, 0)                #目标小于0.5，置1
    pre_zero = np.where(pre <= 0.5, 1, 0)  # 预测大于0.5，置1
    target_zero = np.where(target <= 0.5, 1, 0)  # 目标小于0.5，置1
    pre_one_ = pre_one + target_one
    pre_zero_ = pre_zero + target_zero

    num_pre_one_corr = np.sum(np.where(pre_one_ == 2, 1, 0))   #预测值为1，并且正确
    num_pre_zero_corr = np.sum(np.where(pre_zero_ == 2, 1, 0))  #预测值为0，并且正确

    num_pre_one = np.sum(pre_one)                            #预测值为1
    num_pre_zero = np.sum(pre_zero)                          #预测值为0
    num_target_one = np.sum(target_one)                       #target中值为1
    num_target_zero = np.sum(target_zero)

    recall_one = num_pre_one_corr / num_target_one
    acc_one = num_pre_one_corr / num_pre_one
    recall_zero = num_pre_zero_corr / num_target_zero
    acc_zero = num_pre_zero_corr / num_pre_zero

    return recall_one, acc_one, recall_zero, acc_zero




def mIou(pre_box, target_box, map_size=256, sub_size=32):
    eps = 1e-8
    pre_box = np.transpose(pre_box, [0, 2, 3, 1])     #通道转换
    target_box = np.transpose(target_box, [0, 2, 3, 1])

    mask = np.where(np.abs(target_box[..., 0]) > 1e-4, 1, 0)
    # tt = np.sum(mask)

    x = np.linspace(1, map_size, map_size)            #解码过程
    y = np.linspace(1, map_size, map_size)
    cy, cx = np.meshgrid(x, y)

    pre_cx = cx + pre_box[..., 0] * map_size
    pre_cy = cy + pre_box[..., 1] * map_size
    pre_w = pre_box[..., 2] * sub_size
    pre_h = pre_box[..., 3] * sub_size


    pre_xmin = pre_cx - pre_w / 2
    pre_xmax = pre_cx + pre_w / 2
    pre_ymin = pre_cy - pre_h / 2
    pre_ymax = pre_cy + pre_h / 2

    target_cx = cx + target_box[..., 0] * map_size
    target_cy = cy + target_box[..., 1] * map_size
    target_w = target_box[..., 2] * sub_size
    target_h = target_box[..., 3] * sub_size

    target_xmin = target_cx - target_w / 2
    target_xmax = target_cx + target_w / 2
    target_ymin = target_cy - target_h / 2
    target_ymax = target_cy + target_h / 2            #解码过程

    xmin = np.where(pre_xmin > target_xmin, pre_xmin, target_xmin)
    xmax = np.where(pre_xmax < target_xmax, pre_xmax, target_xmax)
    ymin = np.where(pre_ymin > target_ymin, pre_ymin, target_ymin)
    ymax = np.where(pre_ymax < target_ymax, pre_ymax, target_ymax)

    intra = (xmax - xmin)*(ymax - ymin)
    union = (pre_xmax - pre_xmin)*(pre_ymax - pre_ymin) + \
            (target_xmax - target_xmin)*(target_ymax - target_ymin) - intra
    iou = intra / (union + eps)
    mask_x = np.where(xmin > xmax, 0, 1)
    mask_y = np.where(ymin > ymax, 0, 1)
    iou = iou * mask_x * mask_y * mask
    mIOU = np.sum(iou) / np.sum(mask)
    return mIOU, iou


def confMonitor(pre_iou, pre_conf, thresh):
    num_iou = np.sum(np.where(pre_iou > thresh, 1, 0))
    num_conf = np.sum(np.where(pre_conf > thresh, 1, 0))
    return num_iou, num_conf




if __name__ == '__main__':

  pre_box = np.ones(shape=(8, 4, 256, 256))
  target_box = np.ones(shape=(8, 4, 256, 256))
  m = mIou(pre_box, target_box)