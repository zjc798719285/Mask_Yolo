import torch as th
import numpy as np
def unet_loss(pre_mask, target_mask, pre_box, target_box, pre_conf):

    pre_person = pre_mask[:, 0, :, :]; mask_person = target_mask[:, 0, :, :]
    pre_car = pre_mask[:, 1, :, :]; mask_car = target_mask[:, 1, :, :]
    loss_conf = conf_loss(pre_box, target_box, pre_conf)

    loss_person = focal_loss6(pre_person, mask_person)
    loss_car = focal_loss6(pre_car, mask_car)

    loss_loc = loc_loss(pre_box, target_box)

    return loss_person + loss_car, loss_loc, loss_conf


def r_scale(tensor):
    t = th.where(tensor > 0.5 * th.ones_like(tensor), th.ones_like(tensor), tensor)
    return t


def focal_loss6(pre, target):
    eps = 1e-6
    mask_one = th.where(target > 0.5 * th.ones_like(target), th.ones_like(target), th.zeros_like(target))  #target标注为1
    mask_zero = th.where(target <= 0.5 * th.ones_like(target), th.ones_like(target), th.zeros_like(target)) #target标注为0

    loss_all = target * (r_scale(th.ones_like(pre) - pre)) * th.log(pre + eps) + \
               (r_scale(pre))*(th.ones_like(target) - target) * th.log(th.ones_like(pre) - pre + eps)

    loss_one = -th.sum(loss_all * mask_one) / (th.sum(mask_one))
    loss_zero = -th.sum(loss_all * mask_zero) / (th.sum(mask_zero))

    loss = loss_one + loss_zero
    return loss

def loc_loss(pre, target):
    mask = th.where(th.abs(target) > th.ones_like(target) * 1e-4, th.ones_like(target), th.zeros_like(target))
    loss_tensor = th.abs(pre - target) * mask
    loss = th.sum(loss_tensor) / (th.sum(mask) /4)
    return loss



def conf_loss(pre_box, target_box, pre_conf):
    eps = 1e-6
    # mask_one = th.unsqueeze(th.where(th.abs(target_box[:, 0, ...]) > th.ones_like(target_box[:, 0, ...]) * 1e-4,
    #                         th.ones_like(target_box[:, 0, ...]), th.zeros_like(target_box[:, 0, ...])), 1)
    #
    # mask_zero = th.unsqueeze(th.where(th.abs(target_box[:, 0, ...]) > th.ones_like(target_box[:, 0, ...]) * 1e-4,
    #                          th.zeros_like(target_box[:, 0, ...]), th.ones_like(target_box[:, 0, ...])), 1)

    iou_tensor = get_iou_online(pre_box, target_box, map_size=256, sub_size=32)
    mask_one = th.where(iou_tensor > th.ones_like(iou_tensor)*0.5, th.ones_like(iou_tensor), th.zeros_like(iou_tensor))
    mask_zero = th.where(iou_tensor <= th.ones_like(iou_tensor) * 0.5, th.ones_like(iou_tensor), th.zeros_like(iou_tensor))
    loss_tensor = th.abs(pre_conf - iou_tensor)

    loss_tensor_one = loss_tensor * mask_one
    loss_tensor_zero = loss_tensor * mask_zero

    loss_one = th.sum(loss_tensor_one) / (th.sum(mask_one) + eps)
    loss_zero = th.sum(loss_tensor_zero) / (th.sum(mask_zero) + eps)

    loss = loss_one + loss_zero

    return loss



def get_iou_online(pre, target, map_size=256, sub_size=32):
    eps = 1e-8
    x_np = np.linspace(1, map_size, map_size)  # 解码过程
    y_np = np.linspace(1, map_size, map_size)
    cy_np, cx_np = np.meshgrid(x_np, y_np)
    cx = th.cuda.FloatTensor(cx_np)
    cy = th.cuda.FloatTensor(cy_np)
    pre_cx = cx + pre[:, 0, ...] * map_size
    pre_cy = cy + pre[:, 1, ...] * map_size
    pre_w = pre[:, 2, ...] * sub_size
    pre_h = pre[:, 3, ...] * sub_size

    pre_xmin = pre_cx - pre_w / 2
    pre_xmax = pre_cx + pre_w / 2
    pre_ymin = pre_cy - pre_h / 2
    pre_ymax = pre_cy + pre_h / 2

    target_cx = cx + target[:, 0, ...] * map_size
    target_cy = cy + target[:, 1, ...] * map_size
    target_w = target[:, 2, ...] * sub_size
    target_h = target[:, 3, ...] * sub_size

    target_xmin = target_cx - target_w / 2
    target_xmax = target_cx + target_w / 2
    target_ymin = target_cy - target_h / 2
    target_ymax = target_cy + target_h / 2  # 解码过程

    xmin = th.where(pre_xmin > target_xmin, pre_xmin, target_xmin)
    xmax = th.where(pre_xmax < target_xmax, pre_xmax, target_xmax)
    ymin = th.where(pre_ymin > target_ymin, pre_ymin, target_ymin)
    ymax = th.where(pre_ymax < target_ymax, pre_ymax, target_ymax)

    intra = (xmax - xmin) * (ymax - ymin)
    union = (pre_xmax - pre_xmin) * (pre_ymax - pre_ymin) + \
            (target_xmax - target_xmin) * (target_ymax - target_ymin) - intra
    iou = intra / (union + eps)
    mask_x = th.where(xmin > xmax, th.zeros_like(xmin), th.ones_like(xmin))
    mask_y = th.where(ymin > ymax, th.zeros_like(xmin), th.ones_like(xmin))
    iou = iou * mask_x * mask_y
    iou = th.unsqueeze(iou, 1)
    return iou



