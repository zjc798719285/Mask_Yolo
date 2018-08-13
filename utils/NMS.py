import numpy as np
import torch as th



def mask_nms(mask, box, conf, mask_thresh, conf_thresh, roi_thresh):
    box = box_decoder(box)
    box = np.reshape(box, (-1, 4))
    conf = np.reshape(conf.detach().cpu().numpy(), (-1, 1))
    mask = np.reshape(np.transpose(mask.detach().cpu().numpy()[0, ...], [1, 2, 0])[..., 0], (-1, 1))
    mask = np.where(conf > conf_thresh, 1, 0) * np.where(mask > mask_thresh, 1, 0)
    (non_zero1, non_zero2) = np.nonzero(mask)
    pick_box = box[non_zero1]
    










    box_512 = box_to_512(pick_box)

    return box_512


def box_decoder(pre_box, map_size=256, sub_size=32):
    pre_box = np.transpose(pre_box.detach().cpu().numpy(), [0, 2, 3, 1])
    x = np.linspace(1, map_size, map_size)  # 解码过程
    y = np.linspace(1, map_size, map_size)
    cy, cx = np.meshgrid(x, y)

    pre_cx = cx + pre_box[..., 0] * map_size
    pre_cy = cy + pre_box[..., 1] * map_size
    pre_w = pre_box[..., 2] * sub_size
    pre_h = pre_box[..., 3] * sub_size

    pre_xmin = np.expand_dims((pre_cx - pre_w / 2)[0, ...], 2)/map_size
    pre_xmax = np.expand_dims((pre_cx + pre_w / 2)[0, ...], 2)/map_size
    pre_ymin = np.expand_dims((pre_cy - pre_h / 2)[0, ...], 2)/map_size
    pre_ymax = np.expand_dims((pre_cy + pre_h / 2)[0, ...], 2)/map_size
    de_box = np.concatenate((pre_xmin, pre_xmax, pre_ymin, pre_ymax), axis=2)

    return de_box

def box_to_512(box):
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