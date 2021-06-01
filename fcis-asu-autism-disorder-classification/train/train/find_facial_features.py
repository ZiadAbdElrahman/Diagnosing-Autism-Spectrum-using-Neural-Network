# facial_features
import json, cv2, os, random
import face_recognition, cv2
from PIL import Image, ImageDraw
import numpy as np

with open('split.json', 'r') as fp:
    split = json.load(fp)

if not os.path.isdir('facial_features'):
    os.makedirs('facial_features')
if not os.path.isdir('facial_features/autistic'):
    os.makedirs('facial_features/autistic')
if not os.path.isdir('facial_features/non_autistic'):
    os.makedirs('facial_features/non_autistic')

facial_features_split = {}
for k in split.keys():
    print(k)
    files = split[k]
    c = 0
    new_files = []
    for i, img_dir in enumerate(files):
        if i % 50 == 0:
            print(f'{i} / {len(files)}')
        save_dir = f'facial_features/{img_dir}'[:-4]
        image = face_recognition.load_image_file(img_dir)
        try:
            face_landmarks_list = face_recognition.face_landmarks(image)[0]
        except:
            top, right, bottom, left = 0, image.shape[1], image.shape[0], 0
            face_landmarks_list = face_recognition.face_landmarks(image, face_locations=[[top, right, bottom, left]])[0]
        arr = np.concatenate(list(face_landmarks_list.values()))
        np.save(f'{save_dir}.npy', arr)
        new_files.append(f'{save_dir}.npy')
    facial_features_split[k] = new_files

with open('facial_features_split.json', 'w') as fp:
    json.dump(facial_features_split, fp)
