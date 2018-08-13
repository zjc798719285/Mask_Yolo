import numpy as np
import torch as th



def mask_nms(mask, box, conf, mask_thresh, conf_thresh, roi_thresh):

    MAP_SIZE = 256
    SUB_SIZE = 64

    x = np.linspace(1, MAP_SIZE, MAP_SIZE)  # 解码过程
    y = np.linspace(1, MAP_SIZE, MAP_SIZE)
    cy, cx = np.meshgrid(x, y)
    cx = np.reshape(cx, (-1, 1))
    cy = np.reshape(cy, (-1, 1))

    box = np.reshape(box.detach().cpu().numpy(), (-1, 4))
    conf = np.reshape(conf.detach().cpu().numpy(), (-1, 1))
    mask = np.reshape(np.transpose(mask.detach().cpu().numpy()[0, ...], [1, 2, 0])[..., 0], (-1, 1))
    mask = np.where(conf > conf_thresh, 1, 0) * np.where(mask > mask_thresh, 1, 0)
    (non_zero1, non_zero2) = np.nonzero(mask)
    pick_box = box[non_zero1]
    pick_cx = cx[non_zero1]
    pick_cy = cy[non_zero1]
    pick_box = box_decoder(pre_box=pick_box, cx=pick_cx, cy=pick_cy, map_size=MAP_SIZE, sub_size=SUB_SIZE)




    box_512 = box_to_512(pick_box)

    return box_512


def box_decoder(pre_box, cx, cy, map_size, sub_size):

    pre_cx = cx[:, 0] + pre_box[:, 0] * map_size
    pre_cy = cy[:, 0] + pre_box[:, 1] * map_size
    pre_w = pre_box[..., 2] * sub_size
    pre_h = pre_box[..., 3] * sub_size

    pre_xmin = np.expand_dims((pre_cx - pre_w / 2), 1)/map_size
    pre_xmax = np.expand_dims((pre_cx + pre_w / 2), 1)/map_size
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