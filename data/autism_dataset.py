import os, cv2, json, random
import numpy as np
from typing import Dict, List
from data.base_dataset import BaseDataset, get_params, get_transform


class AutismDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.mode = 'train' if opt.isTrain else 'val'
        if opt.use_augmented_data:
            split_file = f'{self.root}/augmented_split.json'
        else:
            split_file = f'{self.root}/split.json'

        with open(split_file, 'r') as fp:
            self.split = json.load(fp)
        self.files = self.split[self.mode]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict:
        img_dir = f'{self.root}/{self.files[index]}'
        img = cv2.imread(img_dir)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 127.5 - 1
        gray = np.expand_dims(gray, 0)

        label = np.array([int('non' not in img_dir)])
        row = {'A': gray, 'B': label}

        return row
