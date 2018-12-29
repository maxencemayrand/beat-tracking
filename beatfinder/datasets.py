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
                 song_duration,
                 name):
        self.audio_file    = audio_file
        self.beats_file    = beats_file
        self.spec_file     = spec_file
        self.onsets_file   = onsets_file
        self.stretch       = stretch
        self.offset        = offset        # starting point (in seconds) on the stretched wav file.
        self.duration      = duration      # duration in seconds
        self.length        = length        # same as duration, but is samples
        self.song_duration = song_duration # total duration of the stretched wav
        self.name          = name

    def get_wav(self):
        if self.stretch != 1:
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
        all_beats = np.loadtxt(self.beats_file) * self.stretch
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

    def augment(self, stretch_low=2/3, stretch_high=4/3):
        self.stretch = stretch_low + np.random.rand() * (stretch_high - stretch_low)
        self.song_duration *= self.stretch
        self.offset  = np.random.rand() * (self.song_duration - self.duration)

    def correct(self, tightness=500):
        spec, onsets, isbeat, beats = self.get_data()
        corrected_beats  = utils.correct_beats(onsets, beats)
        onsets, isbeat = utils.onsets_and_isbeat(spec, corrected_beats)
        utils.save_onsets_and_isbeat(self.onsets_file, onsets, isbeat)
        
    def predicted_beats(self, tightness=300):
        onsets, isbeat = self.get_onsets_and_isbeat()
        beats_frames = onsets[isbeat == 1]
        if len(beats_frames) == 0 or len(beats_frames) == 1:
            return librosa.frames_to_time(beats_frames, constants.sr, constants.hl), np.nan
        beats, bpm = utils.beat_track(onsets[isbeat == 1], tightness)
        return beats, bpm
    
        
class AudioBeatsDataset(Dataset):

    def __init__(self, audiobeats_list=None, transform=None, file=None):
        if audiobeats_list:
            self.audiobeats_list = audiobeats_list
        elif file:
            self.audiobeats_list = self.load(file)
        self.transform = transform

    def __len__(self):
        return len(self.audiobeats_list)

    def __getitem__(self, i):
        audiobeats = self.audiobeats_list[i]
        if self.transform:
            return self.transform(audiobeats)
        else:
            return audiobeats

    def __add__(self, other):
        return ConcatAudioBeatsDataset([self, other])

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
            print(f' {100*(j+1)/len(self):6.2f}% | ETA: {eta} | {audiobeats.name}' + 20 * ' ', end='\r')

    def save(self, file):
        path = os.path.dirname(file)
        if not os.path.exists(path):
            os.makedirs(path)
        df = pd.DataFrame()
        df['audio_file'] = [self.audiobeats_list[i].audio_file for i in range(len(self))]
        df['beats_file'] = [self.audiobeats_list[i].beats_file for i in range(len(self))]
        df['spec_file'] = [self.audiobeats_list[i].spec_file for i in range(len(self))]
        df['onsets_file'] = [self.audiobeats_list[i].onsets_file for i in range(len(self))]
        df['stretch'] = [self.audiobeats_list[i].stretch for i in range(len(self))]
        df['offset'] = [self.audiobeats_list[i].offset for i in range(len(self))]
        df['duration'] = [self.audiobeats_list[i].duration for i in range(len(self))]
        df['length'] = [self.audiobeats_list[i].length for i in range(len(self))]
        df['song_duration'] = [self.audiobeats_list[i].song_duration for i in range(len(self))]
        df['name'] = [self.audiobeats_list[i].name for i in range(len(self))]
        df.to_csv(file)

    def load(self, file):
        df = pd.read_csv(file, index_col=0)
        audiobeats_list = []
        for i in range(len(df)):
            audio_file    = df['audio_file'][i]
            beats_file    = df['beats_file'][i]
            spec_file     = df['spec_file'][i]
            onsets_file   = df['onsets_file'][i]
            stretch       = df['stretch'][i]
            offset        = df['offset'][i]
            duration      = df['duration'][i]
            length        = df['length'][i]
            song_duration = df['song_duration'][i]
            name          = df['name'][i]
            audiobeats = AudioBeats(audio_file, beats_file, spec_file, onsets_file,
                                    stretch, offset, duration, length, song_duration, name)
            audiobeats_list.append(audiobeats)
        return audiobeats_list

    def augment(self, stretch_low=2/3, stretch_high=4/3):
        for audiobeats in self.audiobeats_list:
            audiobeats.augment(stretch_low, stretch_high)
    
    def correct(self, tightness=500):
        for i, audiobeats in enumerate(self.audiobeats_list):
            audiobeats.correct(tightness)
            print(f'{i+1}/{len(self)}', end='\r')
   
    def clean(self, d=0.08, tightness=300):
        new_list = []
        for i, audiobeats in enumerate(self.audiobeats_list):
            ground_truth    = audiobeats.get_beats()
            correction, bpm = audiobeats.predicted_beats(tightness=tightness)
            F = utils.F_measure(ground_truth, correction, d=d)
            if (F != None) and F > 0.9:
                new_list.append(audiobeats)
            print(f'{i+1}/{len(self)}', end='\r')
        self.audiobeats_list = new_list
            
class SubAudioBeatsDataset(AudioBeatsDataset):

    def __init__(self, dataset, indices):
        audiobeats_list = [dataset.audiobeats_list[i] for i in indices]
        super().__init__(audiobeats_list, dataset.transform)


class AudioBeatsDatasetFromSong(AudioBeatsDataset):

    def __init__(self, audio_file, beats_file, precomputation_path,
                 transform=None, duration=10, stretch=1, force_nb_samples=None):

        self.audio_file = audio_file
        self.song_name = os.path.splitext(os.path.basename(self.audio_file))[0]
        self.song_duration = librosa.get_duration(filename=self.audio_file) * stretch
        self.precomputation_path = precomputation_path

        length = librosa.time_to_samples(duration, constants.sr)


        if force_nb_samples:
            nb_samples = force_nb_samples
        else:
            nb_samples = int(self.song_duration / duration)

        audiobeats_list = []
        for i in range(nb_samples):
            name = f'{self.song_name}.{i:03d}'
            spec_file = os.path.join(self.precomputation_path, f'specs/{name}.npy')
            onsets_file = os.path.join(self.precomputation_path, f'onsets/{name}.csv')
            offset = i * duration
            audiobeats = AudioBeats(audio_file,
                                    beats_file,
                                    spec_file,
                                    onsets_file,
                                    stretch,
                                    offset,
                                    duration,
                                    length,
                                    self.song_duration,
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

    def __init__(self, audio_files, precomputation_path,
                 transform=None, duration=10, stretch=1, force_nb_samples=None):

        dataset_path = os.path.dirname(audio_files)
        beats_dir = os.path.join(dataset_path, 'beats/')

        datasets = []
        with open(audio_files) as f:
            for line in f.readlines():
                relative_audio_file = os.path.normpath(line.strip('\n'))
                audio_name = os.path.splitext(os.path.basename(relative_audio_file))[0]
                audio_file = os.path.join(dataset_path, relative_audio_file)
                beats_file = os.path.join(beats_dir, f'{audio_name}.beats')
                dataset = AudioBeatsDatasetFromSong(audio_file, beats_file, precomputation_path,
                                                    transform, duration, stretch, force_nb_samples)
                datasets.append(dataset)

        super().__init__(datasets)


class BALLROOM(AudioBeatsDatasetFromList):
    def __init__(self, precomputation_path, file_list='allBallroomFiles',
                 transform=None, duration=10, stretch=1, force_nb_samples=None):
        audio_files = './data/BALLROOM/BallroomData/' + file_list
        beats_dir = './data/BALLROOM/beats/'
        super().__init__(audio_files, beats_dir, precomputation_path, transform, duration, stretch, force_nb_samples)


class GTZAN(AudioBeatsDatasetFromList):
    def __init__(self, precomputation_path, file_list='songs_files.txt',
                 transform=None, duration=10, stretch=1, force_nb_samples=None):
        audio_files = './data/GTZAN/' + file_list
        beats_dir = './data/GTZAN/beats/'
        super().__init__(audio_files, beats_dir, precomputation_path, transform, duration, stretch, force_nb_samples)
