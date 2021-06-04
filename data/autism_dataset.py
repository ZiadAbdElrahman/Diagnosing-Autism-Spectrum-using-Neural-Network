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
        elif opt.use_localized_data:
            split_file = f'{self.root}/localized_split.json'
        # elif opt.use_facial_features_data:
        #     split_file = f'facial_features_split'
        else:
            split_file = f'{self.root}/split.json'

        with open(split_file, 'r') as fp:
            self.split = json.load(fp)

        with open(f'{self.root}/facial_features_split.json', 'r') as fp:
            self.facial_features_split = json.load(fp)
            self.facial_features_files = self.facial_features_split[self.mode]

        self.files = self.split[self.mode]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict:
        img_dir = f'{self.root}/{self.files[index]}'

        img = cv2.imread(img_dir)
        # img = cv2.resize(img, (299, 299))
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = np.expand_dims(gray, 2)
        # if self.opt.use_facial_features_data:
        #     points_dir = f'{self.root}/{self.facial_features_files[index]}'
        #     points = np.load(points_dir)
        #     for i in range(len(points)):
        #         point = points[i]
        #         img = cv2.circle(img, (point[0], point[1]), radius=4, color=(255, 255, 255), thickness=-1)
        # # cv2.imshow('g', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # img -> 224 , 244, 3
        # 3 , 244, 244
        img = img / 255
        img = img.transpose(2, 0, 1)

        label = np.array([int('non' not in img_dir)])
        row = {'A': img, 'B': label}

        return row
