import os, random
import json

autistic = os.listdir(f'./autistic')
non_autistic = os.listdir(f'./non_autistic')

autistic = [f'autistic/{f}' for f in autistic]
non_autistic = [f'non_autistic/{f}' for f in non_autistic]

random.shuffle(autistic)
random.shuffle(non_autistic)

num_train = int(len(autistic) * 0.7)

train_aut = autistic[:num_train + 1]
val_aut = autistic[num_train + 1:]

train_non_aut = non_autistic[:num_train + 1]
val_non_aut = non_autistic[num_train + 1:]

train = train_aut + train_non_aut
val = val_aut + val_non_aut

random.shuffle(train)
random.shuffle(val)

split = {'train': train, 'val': val}

with open('split.json', 'w') as fp:
    json.dump(split, fp)

