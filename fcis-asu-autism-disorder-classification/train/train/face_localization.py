# localized
import json, cv2, os, random
import face_recognition, cv2
from PIL import Image, ImageDraw
import numpy as np

with open('split.json', 'r') as fp:
    split = json.load(fp)

if not os.path.isdir('localized'):
    os.makedirs('localized')
if not os.path.isdir('localized/autistic'):
    os.makedirs('localized/autistic')
if not os.path.isdir('localized/non_autistic'):
    os.makedirs('localized/non_autistic')
localized_split = {}

for k in split.keys():
    print(k)
    files = split[k]
    c = 0
    new_files = []
    for i, img_dir in enumerate(files):
        if i % 50 == 0:
            print(f'{i} / {len(files)}')
        save_dir = f'localized/{img_dir}'
        new_files.append(save_dir)

        image = face_recognition.load_image_file(img_dir)
        face_locations = face_recognition.face_locations(image)
        try:
            top, right, bottom, left = face_locations[0]
            new_img = image[top:bottom, left:right]
            v_pad = (image.shape[0] - new_img.shape[0]) // 2
            h_pad = (image.shape[1] - new_img.shape[1]) // 2

            extra_v = 0
            extra_h = 0
            if new_img.shape[0] + v_pad * 2 != image.shape[0]:
                extra_v = 1
            if new_img.shape[1] + h_pad * 2 != image.shape[1]:
                extra_h = 1
            in_image = image
            image = np.pad(new_img, pad_width=[(v_pad, v_pad + extra_v), (h_pad, h_pad + extra_h), (0, 0)],
                           mode='constant')
            assert image.shape == in_image.shape
        except:
            c += 1
            pass
        cv2.imwrite(save_dir, image[:, :, ::-1])
        localized_split[k] = new_files
    print(k, c)

with open('localized_split.json', 'w') as fp:
    json.dump(localized_split, fp)
