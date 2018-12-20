import os
import numpy as np
import pandas as pd
import jams
from constants import *
import librosa
from torch.utils.data import Dataset
import preprocessing

class BeatTrackingDataset(Dataset):
    
    def __init__(self, path, transform=None, sample_length=10):
        self.path = path
        self.specs_path = path + 'specs/'
        self.onsets_path = path + 'onsets/'

        self.sample_length = sample_length
        self.nb_frames_per_sample = librosa.time_to_frames(self.sample_length, sr, hl)
        self.transform = transform
        
    def __len__(self):
        raise NotImplementedError

    def filename(self, i):
        raise NotImplementedError

    def sample_number(self, i):
        raise NotImplementedError
    
    def wav_file(self, i):
        raise NotImplementedError
    
    def annotation_file(self, i):
        raise NotImplementedError

    def load_beats(self, file):
        raise NotImplementedError
        
    def offset(self, i):
        return self.sample_number(i) * self.sample_length
    
    def get_wav(self, i):
        wav = librosa.load(self.wav_file(i), sr=sr, offset=self.offset(i), duration=self.sample_length)[0]
        return wav

    def spec_file(self, i):
        return f'{self.specs_path}{self.filename(i)}.{self.sample_number(i)}.npy'

    def onset_file(self, i):
        return f'{self.onsets_path}{self.filename(i)}.{self.sample_number(i)}.csv'

    def get_beats(self, i):
        beats = self.load_beats(self.annotation_file(i))
        offset = self.offset(i)
        j = self.sample_number(i)
        mask = (offset <= beats) & (beats < offset + self.sample_length)
        beats = beats[mask] - offset
        return beats
        
    def precompute(self):
        for i in range(len(self)):
            spec = preprocessing.get_spec(self.get_wav(i))
            np.save(self.spec_file(i), spec)

            beats = self.get_beats(i)
            onsets, isbeat = preprocessing.get_onsets_from_beats(spec, beats)
            preprocessing.save_onsets(self.onset_file(i), onsets, isbeat)

            print(f'{100*(i+1)/len(self):6.2f}% | {self.filename(i)}.{self.sample_number(i)}')
    
    def __getitem__(self, i):
        spec = np.load(self.spec_file(i))
        df = pd.read_csv(self.onset_file(i))
        onsets = df.onsets.values
        isbeat = df.isbeat.values
        
        if self.transform:
            spec, onsets, isbeat = self.transform((spec, onsets, isbeat))
       
        return spec, onsets, isbeat
    
    
class GTZAN(BeatTrackingDataset):
    
    def __init__(self, transform=None, sample_length=10, nb_files=1000, path='./data/GTZAN/'):
        super().__init__(path, transform, sample_length)
        
        self.wavs_path = path + 'genres/'
        self.annotations_path = path + 'rhythm/'
        
        self.nb_files = nb_files
        self.nb_samples_per_file = int(30 / self.sample_length)
        self.styles = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    def __len__(self):
        return self.nb_files * self.nb_samples_per_file
    
    def get_style_and_number(self, i):
        s = self.styles[i // (100 * self.nb_samples_per_file)]
        n = (i % (100 * self.nb_samples_per_file)) // self.nb_samples_per_file
        return s, n

    def filename(self, i):
        s, n = self.get_style_and_number(i)
        return f'{s}.{n:05}'

    def sample_number(self, i):
        return i % self.nb_samples_per_file
    
    def wav_file(self, i):
        s, n = self.get_style_and_number(i)
        return f'{self.wavs_path}{s}/{self.filename(i)}.au'

    def annotation_file(self, i):
        return f'{self.annotations_path}jams/{self.filename(i)}.wav.jams'

    def load_beats(self, file):
        df = jams.load(file)['annotations'][0].to_dataframe()
        beats = df['time'].values
        return beats


    
class BALLROOM(BeatTrackingDataset):
    
    def __init__(self, transform=None, sample_length=10, nb_files=698, path='./data/BALLROOM/'):
        super().__init__(path, transform, sample_length)

        self.wavs_path = path + 'BallroomData/'
        self.annotations_path = path + 'BallroomAnnotations/'

        self.names = []
        self.wavs_files = []
        with open(self.wavs_path + 'allBallroomFiles') as f:
            for line in f.readlines():
                file = line.strip('\n')[2:]
                self.names.append(os.path.splitext(os.path.basename(file))[0])
                self.wavs_files.append(self.wavs_path + file)
        
        self.nb_files = nb_files
        self.nb_samples_per_file = int(30 / self.sample_length)
     
    def __len__(self):
        return self.nb_files * self.nb_samples_per_file

    def filename(self, i):
        return self.names[i]
    
    def sample_number(self, i):
        return i % self.nb_samples_per_file
    
    def wav_file(self, i):
        return self.wavs_files[i]

    def annotation_file(self, i):
        return f'{self.annotations_path}{self.filename(i)}.beats'

    def load_beats(self, file):
        lines = open(file).readlines()
        beats = np.zeros(len(lines))
        for k in range(len(lines)):
            beats[k] = float(lines[k].split()[0])
        return beats
