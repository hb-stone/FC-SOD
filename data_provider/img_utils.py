import numpy as np
from typing import Tuple
import cv2
import random


def random_scale_image(image:np.ndarray,interpolation=cv2.INTER_LINEAR):
    f_scale = 0.5 + random.randint(0, 10) / 10.0
    image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=interpolation)
    return image

def random_scale_pair_image(image:np.ndarray, label:np.ndarray):
    f_scale = 0.5 + random.randint(0, 10) / 10.0
    image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
    return image, label

def square_crop(img:np.ndarray, label:np.ndarray, need_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    random crop a square img from source and label image
    :param img: source image [H,W,C]
    :param label: label image [H,W,C]
    :param need_size: needed size to scale
    :return:
    """
    assert img.shape[:2] == label.shape[:2]
    height, width = img.shape[0:2]

    if height >= width:
        crop_start = random.randint(0, height - width)
        crop_end = crop_start + width
        img = img[crop_start:crop_end, ...]
        label = label[crop_start:crop_end, ...]
    else:
        crop_start = random.randint(0, width - height)
        crop_end = crop_start + height
        img = img[:, crop_start:crop_end, ...]
        label = label[:, crop_start:crop_end, ...]
    min_size = min(height, width)
    if min_size >= need_size:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR
    img = cv2.resize(img, dsize=(need_size, need_size), interpolation=interpolation)
    label = cv2.resize(label, dsize=(need_size, need_size), interpolation=cv2.INTER_NEAREST)
    return img, label


