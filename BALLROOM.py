import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

ballroom_path = './data/BALLROOM/'
wav_path = ballroom_path + 'BallroomData/'
annotations_path = ballroom_path + 'BallroomAnnotations/'
specs_path = ballroom_path + 'specs/'
onsets_path = ballroom_path + 'onsets/'

names = []
wav_files = []
with open(wav_path + 'allBallroomFiles') as f:
    for line in f.readlines():
        file = line.strip('\n')[2:]
        name = os.path.splitext(os.path.basename(file))[0]
        names.append(name)
        wav_files.append(wav_path + file)

def wav_file(i):
    return wav_files[i]

def annotation_file(i):
    return annotations_path + names[i] + '.beats'

def spec_file(i):
    return specs_path + names[i] + '.npy'

def onset_file(i):
    return onsets_path + names[i] + '.csv'

def load_beats(i):
    beats_df = pd.read_csv(annotation_file(i), sep=' ', header=None)
    return beats_df[0].values

def load_onsets(i):
    df = pd.read_csv(onset_file(i))
    onsets = df.onsets.values
    isbeat = df.isbeat.values
    return onsets, isbeat


class BALLROOM(Dataset):
    
    def __init__(self, size=len(names), transform=None):
        self.size = size
        self.transform = transform
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, i):
        spec = np.load(spec_file(i))
        onsets, isbeat = load_onsets(i)
        if self.transform:
            spec, onsets, isbeat = self.transform((spec, onsets, isbeat))
        return spec, onsets, isbeat










