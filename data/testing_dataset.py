import os, cv2, json, random
import numpy as np
from typing import Dict, List
from data.base_dataset import BaseDataset, get_params, get_transform


class TestingDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.root = self.root.replace('train', 'test')
        print(self.root)
        self.num_imgs = len(os.listdir(self.root))

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index) -> dict:
        img_name = f'{index}.jpg'
        img_dir = f'{self.root}/{img_name}'
        img = cv2.imread(img_dir)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 127.5 - 1
        gray = np.expand_dims(gray, 0)

        # img = img.transpose(2, 0, 1)
        row = {'A': gray, 'name': img_name}

        return row
