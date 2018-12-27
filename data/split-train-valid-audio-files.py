import numpy as np
import os
from sys import argv

dataset_path = argv[1]

audio_files = open(os.path.join(dataset_path, 'audio_files.txt')).readlines()
L = len(audio_files)

p = 0.8
N = int(p * L)
idxs = np.random.permutation(L)
train_audio_files = [audio_files[i] for i in idxs[:N]]
valid_audio_files = [audio_files[i] for i in idxs[N:]]

with open(os.path.join(dataset_path, 'train_audio_files.txt'), 'w') as f:
    f.writelines(train_audio_files)

with open(os.path.join(dataset_path, 'valid_audio_files.txt'), 'w') as f:
    f.writelines(valid_audio_files)
