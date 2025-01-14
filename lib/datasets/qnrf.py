
import os

import cv2
import numpy as np
from PIL import Image
import json
import torch
from torch.nn import functional as F
import random
from .base_dataset import BaseDataset
from .nwpu import NWPU

class QNRF(NWPU):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=1,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=2048,
                 crop_size=(512, 1024),
                 min_unit = (32,32),
                 center_crop_test=False,
                 downsample_rate=1,
                 scale_factor=(0.5, 1/0.5),
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        super(QNRF, self).__init__(
            root,
            list_path,
            num_samples,
            num_classes,
            multi_scale,
            flip,
            ignore_label,
            base_size,
            crop_size,
            min_unit ,
            center_crop_test,
            downsample_rate,
            scale_factor,
            mean,
            std)
    def read_files(self):
        files = []

        if 'test'in self.list_path:
            for item in self.img_list:
                image_id = item[0]
                files.append({
                    "img": 'images/' + image_id + '.jpg',
                    "label": 'jsons/' + image_id + '.json',
                    "name": image_id,
                })

        else:
            for item in self.img_list:
                # import pdb
                # pdb.set_trace()
                image_id = item[0]
                files.append({
                    "img": 'images/' + image_id + '.jpg',
                    "label": 'jsons/' + image_id + '.json',
                    "name": image_id,
                })
        return files