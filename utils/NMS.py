import numpy as np
import torch as th



def mask_nms(mask, box, mask_thresh, e_thresh, roi_thresh):

    box = box_decoder(box)
    box = np.reshape(box, (-1, 8))
    mask = np.reshape(np.transpose(mask.detach().cpu().numpy()[0, ...], [1, 2, 0])[..., 0], (-1, 1))
    mask = np.where(mask > mask_thresh, 1, 0)
    (non_zero1, non_zero2) = np.nonzero(mask)
    pick_box = box[non_zero1]
    if len(pick_box) == 0:
        return []
    pick_box = np.array([i for i in pick_box if i[0] > 0 and i[1] > 0 and i[2] > 0 and i[3] > 0
                                            and i[4] > 0 and i[5] > 0 and i[6] > 0 and i[7] > 0])  #预测有负数的box删除


    e = 0.5*(np.max(pick_box[:, 4:6], axis=1) / np.min(pick_box[:, 4:6], axis=1) +
             np.max(pick_box[:, 6:8], axis=1) / np.min(pick_box[:, 6:8], axis=1))#根据预测出的相对坐标,计算矩形偏心率
    idx_ = np.argsort(e)
    sort_e = e[idx_]
    idx_e = 0
    for i, e_i in enumerate(sort_e):
        idx_e = i
        if e_i > e_thresh:               #根据偏心率删除一部分box
            break
    sort_box = pick_box[idx_[0:idx_e]]
    get_box = []
    while sort_box.shape[0] > 0:
        best_box = sort_box[0]
        sort_box, best_box = del_box(sort_box, best_box, thresh=roi_thresh)
        get_box.append(best_box)

    box_512 = box_to_512(np.array(get_box))
    return box_512


def del_box(sort_box, best_box, thresh):
    eps = 1e-8
    xmin = np.where(sort_box[:, 0] > best_box[0], sort_box[:, 0], best_box[0])
    xmax = np.where(sort_box[:, 1] < best_box[1], sort_box[:, 1], best_box[1])
    ymin = np.where(sort_box[:, 2] > best_box[2], sort_box[:, 2],  best_box[2])
    ymax = np.where(sort_box[:, 3] < best_box[3], sort_box[:, 3],  best_box[3])

    intra = (xmax - xmin)*(ymax - ymin)
    union = (sort_box[:, 1] - sort_box[:, 0]) * (sort_box[:, 3] - sort_box[:, 2]) + \
            (best_box[1] - best_box[0]) * (best_box[3] - best_box[2])
    iou = intra / (union + eps)
    mask_x = np.where(xmin >= xmax, 0, 1)
    mask_y = np.where(ymin >= ymax, 0, 1)
    iou = iou * mask_x * mask_y
    mask_iou = np.where(iou > thresh, 0, 1)
    mask_del = np.ones_like(mask_iou) - mask_iou
    idx_del = np.nonzero(mask_del)
    idx = np.nonzero(mask_iou)
    del_box = sort_box[idx_del]
    best_box = merge_box(del_box, best_box)
    sort_box = sort_box[idx]
    return sort_box, best_box


def merge_box(del_box, best_box):
    xmin = np.max(del_box[:, 0])
    xmax = np.min(del_box[:, 1])
    ymin = np.max(del_box[:, 2])
    ymax = np.min(del_box[:, 3])
    bbox = np.array([xmin, xmax, ymin, ymax])

    return bbox


def box_decoder(pre_box, map_size=128, sub_size=16):
    pre_box = np.transpose(pre_box.detach().cpu().numpy(), [0, 2, 3, 1])
    x = np.linspace(0, map_size-1, map_size)  # 解码过程
    y = np.linspace(0, map_size-1, map_size)
    cy, cx = np.meshgrid(x, y)

    pre_xmin = cx - pre_box[..., 0] * map_size
    pre_xmax = cx + pre_box[..., 1] * map_size
    pre_ymin = cy - pre_box[..., 2] * map_size
    pre_ymax = cy + pre_box[..., 3] * map_size  # 解码过程

    pre_xmin = np.expand_dims(pre_xmin[0, ...], 2)/map_size
    pre_xmax = np.expand_dims(pre_xmax[0, ...], 2)/map_size
    pre_ymin = np.expand_dims(pre_ymin[0, ...], 2)/map_size
    pre_ymax = np.expand_dims(pre_ymax[0, ...], 2)/map_size
    de_box = np.concatenate((pre_xmin, pre_xmax, pre_ymin, pre_ymax, pre_box[0, ...]), axis=2)

    return de_box


def box_to_512(box):
    if len(box) < 1:
        return []
    box = box * 512
    box[..., 0] = np.where(box[..., 0] < 0, 0, box[..., 0])
    box[..., 0] = np.where(box[..., 0] > 511, 511, box[..., 0])

    box[..., 1] = np.where(box[..., 1] < 0, 0, box[..., 1])
    box[..., 1] = np.where(box[..., 1] > 511, 511, box[..., 1])

    box[..., 2] = np.where(box[..., 2] < 0, 0, box[..., 2])
    box[..., 2] = np.where(box[..., 2] > 511, 511, box[..., 2])

    box[..., 3] = np.where(box[..., 3] < 0, 0, box[..., 3])
    box[..., 3] = np.where(box[..., 3] > 511, 511, box[..., 3])
    box = box.astype(np.int16)

    return box