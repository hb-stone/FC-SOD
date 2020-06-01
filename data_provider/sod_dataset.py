import os
import numpy as np
import cv2
from torch.utils import data
import torch
from data_provider.img_utils import square_crop
from data_provider.img_utils import random_scale_pair_image
from typing import Tuple
import random
from copy import deepcopy


class SaliencyDataSet(data.Dataset):


    def __init__(self, root_dir:str, list_path:str, ignore_value:float = 255.0, \
                 crop_size:Tuple[int,int] = (None, None),
                 img_mean:np.ndarray = np.array((104.00699, 116.66877, 122.67892)), \
                 is_random_flip:bool = False, is_scale:bool = False ) -> None:
        img_ids = [i_id.strip() for i_id in open(list_path) if i_id.strip() != '']
        self.files = []
        self.img_mean = img_mean
        for img_id in img_ids:
            img_name, gt_name = img_id.split()
            img_file = os.path.join(root_dir, img_name)
            label_file = os.path.join(root_dir, gt_name)
            self.files.append({
                "img": img_file,
                "label": label_file,
            })
        self.files.sort(key=lambda x:x['img'])
        self.is_random_flip:bool = is_random_flip
        self.is_scale:bool = is_scale
        self.ignore_value:float = ignore_value
        if crop_size is not None and crop_size != (0,0):
            self.crop_height,self.crop_width = crop_size
        else:
            self.crop_height, self.crop_width = None,None


    def __iadd__(self, other):
        # not verify other options
        self.files.extend(other.files)
        return self

    def __add__(self, other):
        ret = deepcopy(self)
        ret.files.extend(other.files)
        return ret


    def __len__(self):
        return len(self.files)

    def __getitem__( self, index: int or slice ):
        if isinstance(index,slice):
            ret = deepcopy(self)
            ret.files = ret.files[index]
            return ret
        elif isinstance(index, list):
            ret = deepcopy(self)
            ret.files = [ret.files[i] for i in index]
            return ret
        datafiles = self.files[index]
        filename = os.path.basename(datafiles["img"])
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        if self.is_scale:
            image, label = random_scale_pair_image(image, label)
        if self.crop_height is not None:
            image, label = square_crop(image, label, self.crop_height)
        image = image.astype(np.float32)
        image -= self.img_mean
        image = image.transpose((2, 0, 1))
        label = label.astype(np.float32)
        label /= 255

        if self.is_random_flip:
            if random.randint(0,1) == 1:
                image = image[..., ::-1]
                label = label[..., ::-1]
        return {
            'image':image.copy(),
            'label':label.copy(),
            'filename':filename
        }


    def real_image(self, image:torch.Tensor) -> torch.Tensor:
        """
        recover the real image from tensor
        :param image:
        :return: the real image value with shape [N,C,H,W]
        """
        # print(image.shape)
        image += torch.from_numpy(self.img_mean).to(image.dtype).view(1,-1,1,1).float()
        image = image.byte()
        return image

    def real_label(self, label:torch.Tensor) -> torch.Tensor:
        """
        return the real image value with shape [C,H,W]
        :param label: the label shape must be [N,H,W] or [N,C,H,W]
        :return:
        """
        ignore_index = (label == self.ignore_value)
        label *= 255.0
        label[ignore_index] == 0.0
        if len(label.shape) == 3:
            label = label.unsqueeze(dim=1)
        if label.size(1) == 1:
            shape = list(label.shape)
            shape[1] = 3
            label = label.expand(shape)
        label = label.byte()
        return label


