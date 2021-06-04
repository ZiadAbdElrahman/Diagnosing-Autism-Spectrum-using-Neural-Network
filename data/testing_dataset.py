import os, cv2, json, random
import numpy as np
from typing import Dict, List
import face_recognition
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

        if self.opt.use_facial_features_data:
            points = self.detect_face_features(img_dir)
            for i in range(len(points)):
                point = points[i]
                img = cv2.circle(img, (point[0], point[1]), radius=4, color=(255, 255, 255), thickness=-1)

        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 127.5 - 1
        # gray = np.expand_dims(gray, 0)
        img = img / 127.5 - 2
        img = img.transpose(2, 0, 1)
        row = {'A': img, 'name': img_name}

        return row

    def detect_face_features(self, img_dir):
        image = face_recognition.load_image_file(img_dir)
        try:
            face_landmarks_list = face_recognition.face_landmarks(image)[0]
        except:
            top, right, bottom, left = 0, image.shape[1], image.shape[0], 0
            face_landmarks_list = face_recognition.face_landmarks(image, face_locations=[[top, right, bottom, left]])[0]
        arr = np.concatenate(list(face_landmarks_list.values()))
        return arr
