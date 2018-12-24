import os
import numpy as np
import pandas as pd
import jams
from constants import *
import librosa
from torch.utils.data import Dataset, ConcatDataset
import preprocessing
import bisect

class BeatTrackingDataset(Dataset):
    
    def __init__(self, transform=None, sample_length=10):
        self.transform = transform        
        self.sample_length = sample_length
        self.wav_length = librosa.time_to_samples(self.sample_length, sr)
        
    def __len__(self):
        raise NotImplementedError
    
    def song_name(self, i):
        raise NotImplementedError
    
    def sample_number(self, i):
        raise NotImplementedError
        
    def wav_file(self, i):
        raise NotImplementedError
    
    def offset(self, i):
        raise NotImplementedError
    
    def get_beats(self, i):
        raise NotImplementedError
        
    def get_wav(self, i):
        raise NotImplementedError

    def precomputation_directory(self, i):
        raise NotImplementedError
        
    def specs_path(self, i):
        path = f'{self.precomputation_directory(i)}specs/'
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    
    def onsets_path(self, i):
        path = f'{self.precomputation_directory(i)}onsets/'
        if not os.path.exists(path):
            os.makedirs(path)
        return path
        
    def spec_file(self, i):
        return f'{self.specs_path(i)}{self.song_name(i)}.{self.sample_number(i):03}.npy'

    def onset_file(self, i):
        return f'{self.onsets_path(i)}{self.song_name(i)}.{self.sample_number(i):03}.csv'
        
    def precompute(self, idxs=None):
        if idxs == None:
            idxs = range(len(self))
        for j, i in enumerate(idxs):
            spec = preprocessing.get_spec(self.get_wav(i))
            np.save(self.spec_file(i), spec)
            beats = self.get_beats(i)
            onsets, isbeat = preprocessing.get_onsets_from_beats(spec, beats)
            preprocessing.save_onsets(self.onset_file(i), onsets, isbeat)
            print(f'{100*(j+1)/len(idxs):6.2f}% | {self.song_name(i)}.{self.sample_number(i):03}')
    
    def __getitem__(self, i):
        spec = np.load(self.spec_file(i))
        df = pd.read_csv(self.onset_file(i))
        onsets = df.onsets.values
        isbeat = df.isbeat.values
        
        if self.transform:
            spec, onsets, isbeat = self.transform((spec, onsets, isbeat))
       
        return spec, onsets, isbeat
    
    def __add__(self, other):
        return ConcatBeatTrackingDataset([self, other])
    
    def set_transform(self, transform):
        self.transform = transform
    
    
class SingleSongBeatTrackingDataset(BeatTrackingDataset):
    
    def __init__(self, 
                 song_file, beats_file, precomputation_path, 
                 transform=None, sample_length=10, stretch=None, force_nb_samples=None):
        super().__init__(transform, sample_length)
        self.song_file = song_file
        self.beats_file = beats_file
        self.precomputation_path = precomputation_path
        self.stretch = stretch
        if self.stretch:
            self.song_duration = librosa.get_duration(filename=self.song_file) * self.stretch
        else:
            self.song_duration = librosa.get_duration(filename=self.song_file)
        self.force_nb_samples = force_nb_samples
        
    def __len__(self):
        if self.force_nb_samples:
            return self.force_nb_samples
        else:
            return int(self.song_duration / self.sample_length)
    
    def song_name(self, i):
        if not (0 <= i < len(self)):
            raise IndexError("Index out of range")
        return os.path.splitext(os.path.basename(self.song_file))[0]
    
    def sample_number(self, i):
        if not (0 <= i < len(self)):
            raise IndexError("Index out of range")
        return i

    def wav_file(self, i):
        if not (0 <= i < len(self)):
            raise IndexError("Index out of range")
        return self.song_file

    def offset(self, i):
        if not (0 <= i < len(self)):
            raise IndexError("Index out of range")
        return i * self.sample_length

    def get_beats(self, i):
        if self.stretch:
            all_beats = np.loadtxt(self.beats_file) * self.stretch
        else:
            all_beats = np.loadtxt(self.beats_file)
        offset = self.offset(i)
        mask = (offset <= all_beats) & (all_beats < offset + self.sample_length)
        beats = all_beats[mask] - offset
        return beats
    
    def get_wav(self, i):
        if self.stretch:
            offset = self.offset(i) / self.stretch
            duration = self.sample_length / self.stretch
            unstretched_wav = librosa.load(self.song_file, sr=sr, offset=offset, duration=duration)[0]
            wav = librosa.effects.time_stretch(unstretched_wav, rate=1/self.stretch)
        else:
            wav = librosa.load(self.song_file, sr=sr, offset=self.offset(i), duration=self.sample_length)[0]
        
        if len(wav) != self.wav_length:
            z = np.zeros(self.wav_length)
            m = min(len(wav), self.wav_length)
            z[:m] = wav[:m]
            wav = z
        
        return wav
    
    def precomputation_directory(self, i):
        if not (0 <= i < len(self)):
            raise IndexError("Index out of range")
        return self.precomputation_path
    
    def set_transform(self, transform):
        self.transform = transform
        
        
class ConcatBeatTrackingDataset(BeatTrackingDataset):

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r
    
    def __init__(self, datasets):
        for i in range(1, len(datasets)):
            if datasets[i].transform != datasets[i - 1].transform:
                raise ValueError("All transforms must be the same")
            if datasets[i].sample_length != datasets[i - 1].sample_length:
                raise ValueError("All sample lengths must be the same")
        
        super().__init__(datasets[0].transform, datasets[0].sample_length)
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]
        
    def dataset_idx(self, i):
        return bisect.bisect_right(self.cumulative_sizes, i)
    
    def sample_idx(self, i):
        ds_idx = self.dataset_idx(i)
        if ds_idx == 0:
            sm_idx = i
        else:
            sm_idx = i - self.cumulative_sizes[ds_idx - 1]
        return sm_idx
    
    def __getitem__(self, i):
        return self.datasets[self.dataset_idx(i)][self.sample_idx(i)]
    
    def sample_number(self, i):
        return self.datasets[self.dataset_idx(i)].sample_number(self.sample_idx(i))
    
    def song_name(self, i):
        return self.datasets[self.dataset_idx(i)].song_name(self.sample_idx(i))
        
    def wav_file(self, i):
        return self.datasets[self.dataset_idx(i)].wav_file(self.sample_idx(i))
    
    def offset(self, i):
        return self.datasets[self.dataset_idx(i)].offset(self.sample_idx(i))
    
    def get_beats(self, i):
        return self.datasets[self.dataset_idx(i)].get_beats(self.sample_idx(i))
        
    def get_wav(self, i):
        return self.datasets[self.dataset_idx(i)].get_wav(self.sample_idx(i))

    def precomputation_directory(self, i):
        return self.datasets[self.dataset_idx(i)].precomputation_directory(self.sample_idx(i))
    
    def set_transform(self, transform):
        self.transform = transform
        for d in self.datasets:
            d.set_transform(transform)
    
    
class BeatTrackingDatasetFromList(ConcatBeatTrackingDataset):
    
    def __init__(self, 
                 songs_files, 
                 beats_dir, 
                 precomputation_path, 
                 transform=None, 
                 sample_length=10, 
                 stretch=None,
                 force_nb_samples=None):
        
        directory = os.path.dirname(songs_files)
        
        datasets = []
        with open(songs_files) as f:
            for line in f.readlines():
                relative_song_file = line.strip('\n')
                song_name = os.path.splitext(os.path.basename(relative_song_file))[0]
                song_file = './' + os.path.normpath(os.path.join(directory, relative_song_file))
                beats_file = os.path.join(beats_dir, f'{song_name}.beats')
                dataset = SingleSongBeatTrackingDataset(song_file,
                                                        beats_file, 
                                                        precomputation_path,
                                                        transform,
                                                        sample_length=sample_length,
                                                        stretch=stretch,
                                                        force_nb_samples=force_nb_samples)
                datasets.append(dataset)
        
        super().__init__(datasets)
                
class BALLROOM(BeatTrackingDatasetFromList):
    def __init__(self, precomputation_path, transform=None, sample_length=10, stretch=None, force_nb_samples=None):
        songs_files = './data/BALLROOM/BallroomData/allBallroomFiles'
        beats_dir = './data/BALLROOM/beats/'
        super().__init__(songs_files, beats_dir, precomputation_path, transform, sample_length, stretch, force_nb_samples)
        
class GTZAN(BeatTrackingDatasetFromList):
    def __init__(self, precomputation_path, transform=None, sample_length=10, stretch=None, force_nb_samples=None):
        songs_files = './data/GTZAN/songs_files.txt'
        beats_dir = './data/GTZAN/beats/'
        super().__init__(songs_files, beats_dir, precomputation_path, transform, sample_length, stretch, force_nb_samples)
    

class SubBeatTrackingDataset(BeatTrackingDataset):
    def __init__(self, dataset, indices):
        super().__init__(dataset.transform, dataset.sample_length)
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        return self.dataset[self.indices[i]]
    
    def song_name(self, i):
        return self.dataset.song_name(self.indices[i])
    
    def sample_number(self, i):
        return self.dataset.sample_number(self.indices[i])
        
    def wav_file(self, i):
        return self.dataset.wav_file(self.indices[i])
    
    def offset(self, i):
        return self.dataset.offset(self.indices[i])
    
    def get_beats(self, i):
        return self.dataset.get_beats(self.indices[i])
        
    def get_wav(self, i):
        return self.dataset.get_wav(self.indices[i])

    def precomputation_directory(self, i):
        return self.dataset.precomputation_directory(self.indices[i])
    
    def set_transform(self, transform):
        self.transform = transform
        self.dataset.set_transform(transform)
    
    
    
    
    
    
    
    
    
    
    
    
        
