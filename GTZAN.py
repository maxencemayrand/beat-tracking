import numpy as np
import pandas as pd
import jams
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

def load_beats(s, n):
    df = jams.load(filename_rhythm(s, n))['annotations'][0].to_dataframe()
    return df['time'].values

def loadGTZAN(s, n, getbeats=False):
    spec = np.load(filename_spec(s, n))
    df = pd.read_csv(filename_onset(s, n))
    onsets = df.onsets.values
    isbeat = df.isbeat.values
    if getbeats:
        return spec, onsets, isbeat, load_beats(s, n)
    else:
        return spec, onsets, isbeat

class StyleGTZAN(Dataset):
    def __init__(self, style):
        self.style = style
    def __len__(self):
        return 100
    def __getitem__(self, n):
        return loadGTZAN(self.style, n)
    
class GTZAN(Dataset):
    def __init__(self, nf=936, style=None, idxs=None, transform=None, getbeats=False):
        """GTZAN dataset (spectrograms, onsets, and onsets selected as first beats).
        arguments:
            style: If `None`, loads the full dataset (idxs ignored). Else, `style` is a string (e.g. 'jazz').
             idxs: If `style != None` will load the files in `style` of indices in `idxs`.
        """
        self.nf = nf
        self.style = style
        if type(idxs) == int:
            self.idxs = range(idxs)
        else:
            self.idxs = idxs
        self.getbeats = getbeats
        self.transform = transform
    
    def __len__(self):
        if self.style == None:
            return 10 * 100  # 10 genres with 100 songs each
        else:
            return len(self.idxs)
    
    def __getitem__(self, i):
        if self.style == None:
            s = styles[i // 100]
            n = i % 100
        else:
            s = self.style
            n = self.idxs[i]
        
        if self.getbeats:
            spec, onsets, isbeat, beats = loadGTZAN(s, n, True)
        else:
            spec, onsets, isbeat = loadGTZAN(s, n, False)
        
        if self.nf != None:
            spec = spec[:, :self.nf]
            mask = onsets < self.nf
            onsets = onsets[mask]
            isbeat = isbeat[mask]
        
        if self.getbeats:
            sample = (spec, onsets, isbeat, beats)
        else:
            sample = (spec, onsets, isbeat)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

    
    
    
    
    
    
    
    
    
    