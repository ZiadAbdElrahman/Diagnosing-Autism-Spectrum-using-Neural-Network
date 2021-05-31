import json, cv2, os, random
from scipy import ndimage

with open('split.json', 'r') as fp:
    split = json.load(fp)

if not os.path.isdir('augmented'):
    os.makedirs('augmented')
if not os.path.isdir('augmented/autistic'):
    os.makedirs('augmented/autistic')
if not os.path.isdir('augmented/non_autistic'):
    os.makedirs('augmented/non_autistic')
augmented_split = {'val': split['val']}

train = split['train']
train_rot45 = []
train_rot_45 = []
train_flip = []
train_flip_rot45 = []
train_flip_rot_45 = []

for i, img_dir in enumerate(train):
    if i % 50 == 0:
        print(f'{i} / {len(train)}')
    img_name = img_dir[:-4]
    save_dir = f'augmented/{img_name}'

    img = cv2.imread(img_dir)
    img_rot45 = ndimage.rotate(img, 45)[47:-46, 47:-46, :]
    img_rot_45 = ndimage.rotate(img, -45)[47:-46, 47:-46, :]
    img_flip = cv2.flip(img, 1)
    img_flip_rot45 = ndimage.rotate(img_flip, 45)[47:-46, 47:-46, :]
    img_flip_rot_45 = ndimage.rotate(img_flip, -45)[47:-46, 47:-46, :]
    cv2.imwrite(f'{save_dir}_rot45.jpg', img_rot45)
    train_rot45.append(f'{save_dir}_rot45.jpg')

    cv2.imwrite(f'{save_dir}_rot_45.jpg', img_rot_45)
    train_rot_45.append(f'{save_dir}_rot_45.jpg')

    cv2.imwrite(f'{save_dir}_flip.jpg', img_flip)
    train_flip.append(f'{save_dir}_flip.jpg')

    cv2.imwrite(f'{save_dir}_flip_rot45.jpg', img_flip_rot45)
    train_flip_rot45.append(f'{save_dir}_flip_rot45.jpg')

    cv2.imwrite(f'{save_dir}_flip_rot_45.jpg', img_flip_rot_45)
    train_flip_rot45.append(f'{save_dir}_flip_rot_45.jpg')
#
augmented_train = train + train_rot45 + train_rot_45 + train_flip + train_flip_rot45 + train_flip_rot_45
random.shuffle(augmented_train)

augmented_split['train'] = augmented_train
with open('augmented_split.json', 'w') as fp:
    json.dump(augmented_split, fp)
