import os
import numpy as np
import pandas as pd
import librosa
from torch.utils.data import Dataset
import bisect
import time
import datetime

from . import utils
from . import constants

class AudioBeats(object):
    def __init__(self, 
                 audio_file, 
                 beats_file, 
                 spec_file,
                 onsets_file,
                 stretch,
                 offset,
                 duration,
                 length,
                 name):
        self.audio_file  = audio_file
        self.beats_file  = beats_file
        self.spec_file   = spec_file
        self.onsets_file = onsets_file
        self.stretch     = stretch
        self.offset      = offset     # starting point (in seconds) on the stretched wav file.
        self.duration    = duration   # duration in seconds
        self.length      = length     # same as duration, but is samples
        self.name        = name

    def get_wav(self):
        if self.stretch:
            offset = self.offset / self.stretch
            duration = self.duration / self.stretch
            unstretched_wav = librosa.load(self.audio_file, sr=constants.sr, offset=offset, duration=duration)[0]
            wav = librosa.effects.time_stretch(unstretched_wav, rate=1/self.stretch)
        else:
            wav = librosa.load(self.audio_file, sr=constants.sr, offset=self.offset, duration=self.duration)[0]
        
        if len(wav) != self.length:
            z = np.zeros(self.length)
            m = min(len(wav), self.length)
            z[:m] = wav[:m]
            wav = z
        
        return wav
    
    def get_beats(self):
        all_beats = np.loadtxt(self.beats_file)
        if self.stretch:
            all_beats *= self.stretch
        mask = (self.offset <= all_beats) & (all_beats < self.offset + self.duration)
        beats = all_beats[mask] - self.offset
        return beats
    
    def precompute_spec(self):
        path = os.path.dirname(self.spec_file)
        if not os.path.exists(path):
            os.makedirs(path)
        spec = utils.spectrogram(self.get_wav())
        np.save(self.spec_file, spec)

    def get_spec(self):
        return np.load(self.spec_file)
    
    def precompute_onsets_and_isbeat(self):
        path = os.path.dirname(self.onsets_file)
        if not os.path.exists(path):
            os.makedirs(path)
        beats = self.get_beats()
        spec = self.get_spec()
        onsets, isbeat = utils.onsets_and_isbeat(spec, beats)
        utils.save_onsets_and_isbeat(self.onsets_file, onsets, isbeat)
    
    def get_onsets_and_isbeat(self):
        df = pd.read_csv(self.onsets_file)
        onsets = df.onsets.values
        isbeat = df.isbeat.values
        return onsets, isbeat
    
    def precompute(self):
        self.precompute_spec()
        self.precompute_onsets_and_isbeat()
    
    def get_data(self):
        spec  = self.get_spec()
        onsets, isbeat = self.get_onsets_and_isbeat()
        beats = self.get_beats()
        return spec, onsets, isbeat, beats
    
    
class AudioBeatsDataset(Dataset):
    
    def __init__(self, audiobeats_list, transform=None):
        self.audiobeats_list = audiobeats_list
        self.transform = transform
    
    def __len__(self):
        return len(self.audiobeats_list)
    
    def __getitem__(self, i):
        audiobeats = self.audiobeats_list[i]
        if self.transform:
            return self.transform(audiobeats)
        else:
            return audiobeats

    def precompute(self, mode='all'):
        for j, audiobeats in enumerate(self):
            t = time.time()
            if mode == 'all':
                audiobeats.precompute()
            elif mode == 'spec':
                audiobeats.precompute_spec()
            elif mode == 'onsets_and_isbeat':
                audiobeats.precompute_onsets_and_isbeat()
            else:
                raise ValueError('Unknown mode')
            t = time.time() - t
            eta = str(datetime.timedelta(seconds=int(t * (len(self) - j - 1))))
            print(f'\r{100*(j+1)/len(self):6.2f}% | ETA: {eta} | {audiobeats.name}' + 20 * ' ', end='')

            
class SubAudioBeatsDataset(AudioBeatsDataset):
    
    def __init__(self, dataset, indices):
        audiobeats_list = [dataset.audiobeats_list[i] for i in indices]
        super().__init__(audiobeats_list, dataset.transform)
        
            
class AudioBeatsDatasetFromSong(AudioBeatsDataset):
    
    def __init__(self, audio_file, beats_file, precomputation_path,
                 transform=None, duration=10, stretch=None, force_nb_samples=None):
        
        self.audio_file = audio_file
        self.song_name = os.path.splitext(os.path.basename(self.audio_file))[0]
        self.song_duration = librosa.get_duration(filename=self.audio_file)
        if stretch:
            self.song_duration *= stretch
        self.precomputation_path = precomputation_path
        
        length = librosa.time_to_samples(duration, constants.sr)

            
        if force_nb_samples:
            nb_samples = force_nb_samples
        else:
            nb_samples = int(self.song_duration / duration)

        audiobeats_list = []
        for i in range(nb_samples):
            name = f'{self.song_name}.{i:03}'
            spec_file = f'{self.precomputation_path}specs/{name}.npy'
            onsets_file = f'{self.precomputation_path}onsets/{name}.csv'
            offset = i * duration
            audiobeats = AudioBeats(audio_file, 
                                    beats_file, 
                                    spec_file,
                                    onsets_file,
                                    stretch,
                                    offset,
                                    duration,
                                    length,
                                    name)
            audiobeats_list.append(audiobeats)
        
        super().__init__(audiobeats_list, transform)
        
        
class ConcatAudioBeatsDataset(AudioBeatsDataset):

    def __init__(self, datasets):
        for i in range(1, len(datasets)):
            if datasets[i].transform != datasets[i - 1].transform:
                raise ValueError("All transforms must be the same")
        
        audiobeats_list = []
        for dataset in datasets:
            audiobeats_list += dataset.audiobeats_list
        
        super().__init__(audiobeats_list, datasets[0].transform)

        
class AudioBeatsDatasetFromList(ConcatAudioBeatsDataset):
    
    def __init__(self, audio_files, beats_dir, precomputation_path, 
                 transform=None, duration=10, stretch=None, force_nb_samples=None):
        
        directory = os.path.dirname(audio_files)
        
        datasets = []
        with open(audio_files) as f:
            for line in f.readlines():
                relative_audio_file = line.strip('\n')
                audio_name = os.path.splitext(os.path.basename(relative_audio_file))[0]
                audio_file = './' + os.path.normpath(os.path.join(directory, relative_audio_file))
                beats_file = os.path.join(beats_dir, f'{audio_name}.beats')
                dataset = AudioBeatsDatasetFromSong(audio_file, beats_file, precomputation_path,
                                                    transform, duration, stretch, force_nb_samples)
                datasets.append(dataset)
        
        super().__init__(datasets)

        
class BALLROOM(AudioBeatsDatasetFromList):
    def __init__(self, precomputation_path, transform=None, duration=10, stretch=None, force_nb_samples=None):
        audio_files = './data/BALLROOM/BallroomData/allBallroomFiles'
        beats_dir = './data/BALLROOM/beats/'
        super().__init__(audio_files, beats_dir, precomputation_path, transform, duration, stretch, force_nb_samples)
            
            
class GTZAN(AudioBeatsDatasetFromList):
    def __init__(self, precomputation_path, transform=None, duration=10, stretch=None, force_nb_samples=None):
        audio_files = './data/GTZAN/songs_files.txt'
        beats_dir = './data/GTZAN/beats/'
        super().__init__(audio_files, beats_dir, precomputation_path, transform, duration, stretch, force_nb_samples)
            
