import numpy as np
import pandas as pd
from torch.utils.data import Dataset

gtzan_path = './data/GTZAN/'
genres_path = gtzan_path + 'genres/'
rhythm_path = gtzan_path + 'rhythm/'
onsets_path = gtzan_path + 'onsets/'
specs_path = gtzan_path + 'specs/'

styles = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def filename(style, number):
    return f'{style}.{number:05}'

def filename_audio(style, number):
    return f'{genres_path}{style}/{filename(style, number)}.au'

def filename_onset(style, number):
    return f'{onsets_path}{filename(style, number)}.csv'

def filename_rhythm(style, number):
    return f'{rhythm_path}jams/{filename(style, number)}.wav.jams'

def filename_spec(style, number):
    return f'{specs_path}{filename(style, number)}.npy'

def loadGTZAN(s, n):
    spec = np.load(filename_spec(s, n))
    df = pd.read_csv(filename_onset(s, n))
    onsets = df.onsets.values
    isbeat = df.isbeat.values
    return spec, onsets, isbeat

class StyleGTZAN(Dataset):
    def __init__(self, style):
        self.style = style
    def __len__(self):
        return 100
    def __getitem__(self, n):
        return loadGTZAN(self.style, n)
    
class GTZAN(Dataset):
    def __init_(self):
        pass
    def __len__(self):
        return 10 * 100  # 10 genres with 100 songs each
    def __getitem__(self, i):
        s = styles[i // 100]
        n = i % 100
        return loadGTZAN(s, n)
