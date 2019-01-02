import os
import numpy as np
import pandas as pd
import librosa
from torch.utils.data import Dataset
import time
import datetime

from . import utils
from . import constants

class AudioBeats(object):
    r"""Basic class to represent a sample point.

    An `AudioBeats` object doesn't contain any data, but points to the right
    files and has many useful methods to precompute some of the data. They are
    meant to be the items of the pytorch Dataset class `AudioBeatsDataset` to
    train the `BeatFinder` model.

    An `AudioBeats` object represents a section of an audio file (possibly
    stretched) together with the beats (a sequence of times), the onsets, the
    subset of those onsets which are beats, and the spectrogram of the audio.
    The method `precompute` computes the spetrograms, onsets, and the onsets
    that are beats and store this information in files so that it can be quickly
    accessed during training.

    Arguments:
        audio_file (str): The relative path of the full audio file.
        beats_file (str): The relative path of the files containing the beats of
            `audio_file`. This is a `txt` file containing a single column of
            floating point numbers which are the times of the beat track (ground
            truth).
        spec_file (str): The relative path of the file where the spectrogram
            is stored upon calling `precompute_spec()`. This should be a `.npy`
            file. The enclosing directory will be created if it doesn't already
            exists.
        onsets_file (str): The relative path of the file which stores the onsets
            (in units of frames) together with whether they are beats. This is
            a `.csv` file with two columns: the onsets' frames and a column of
            0s and 1s.
        stretch (float): The amount by which to stretch the audio (without
            changing the pitch). For example, if `stretch=2`, the AudioBeats
            object represents a section of the same audio file, but twice
            slower.
        offset (float): The starting point of the sample in the stretched audio
            file.
        duration (float): The duration of the audio in seconds.
        length (int): The duration of the audio in samples (at the sampling rate
            determined in `constants.py`).
        song_duration (float): The duration (in seconds) of the full audio file
            after stretching.
        name (str): The name of the AudioBeats object.
    """

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
        r"""Returns a numpy array of the audio section at the sampling rate
        determined by the `constants` module."""

        # Only computes the streching if `self.stretch` is different than 1 as
        # otherwise this is unnecessarily slow when we don't want stretching.
        if self.stretch != 1:
            offset = self.offset / self.stretch
            duration = self.duration / self.stretch
            unstretched_wav = librosa.load(self.audio_file,
                                sr=constants.sr,
                                offset=offset,
                                duration=duration)[0]
            wav = librosa.effects.time_stretch(unstretched_wav,
                                rate=1/self.stretch)
        else:
            wav = librosa.load(self.audio_file,
                    sr=constants.sr,
                    offset=self.offset,
                    duration=self.duration)[0]

        # This is to make sure that all samples have the same size so that we
        # can do minibatches.
        if len(wav) != self.length:
            z = np.zeros(self.length)
            m = min(len(wav), self.length)
            z[:m] = wav[:m]
            wav = z

        return wav

    def get_beats(self):
        r"""Returns a numpy array of the beats in seconds.
        """

        all_beats = np.loadtxt(self.beats_file) * self.stretch
        mask = (self.offset <= all_beats) & (all_beats < self.offset + self.duration)
        beats = all_beats[mask] - self.offset
        return beats

    def precompute_spec(self):
        r"""Compute the mel-scaled spectrograms and store it in
        `self.spec_file`.
        """

        path = os.path.dirname(self.spec_file)
        if not os.path.exists(path):
            os.makedirs(path)
        spec = utils.spectrogram(self.get_wav())
        np.save(self.spec_file, spec)

    def get_spec(self):
        r"""Returns the mel-scaled spectrogram (it must have been precomputed
        beforehand by calling `precompute_spec()` or `precompute()`)."""

        return np.load(self.spec_file)

    def precompute_onsets_and_isbeat(self, full=False):
        r"""Computes the onsets from the spectrogram and select which ones are
        beats. Then stores the result in `self.onsets_file`. Only works if
        `precompute_spec()` has been called before.
        """

        # The directory containing the file is created if it doesn't already
        # exist.
        path = os.path.dirname(self.onsets_file)
        if not os.path.exists(path):
            os.makedirs(path)

        # Get the beats (ground truth) and the spectrogram.
        beats = self.get_beats()
        spec = self.get_spec()

        if full == False:
            # This utils function compute the onsets from the spectrogram and
            # select which ones are beats (within a small interval).
            onsets, isbeat = utils.onsets_and_isbeat(spec, beats)
        else:
            onsets = np.arange(spec.shape[1])
            isbeat = np.zeros_like(onsets)
            if len(beats) != 0:
                times = librosa.frames_to_time(onsets, constants.sr, constants.hl)
                dists = np.min(np.abs(times[:, np.newaxis] - beats), axis=1)
                isbeat[dists < constants.dt] = 1
        
        # Save on `self.onsets_file`.
        utils.save_onsets_and_isbeat(self.onsets_file, onsets, isbeat)

    def get_onsets_and_isbeat(self):
        r"""Returns the onsets (in frames) and an array of 0/1 saying whether
        each onset is a beat. The method `precompute_onsets_and_isbeat` must
        have been called at some point before.
        """

        df = pd.read_csv(self.onsets_file)
        onsets = df.onsets.values
        isbeat = df.isbeat.values
        return onsets, isbeat

    def precompute(self):
        r"""Precomputes the spectrogram, the onsets, and which onsets are beats.
        """

        self.precompute_spec()
        self.precompute_onsets_and_isbeat()

    def get_data(self):
        r"""Returns the spectrogram, the onsets (in frames), which onsets are
        beats, and all the beats (in seconds).

        Returns:
            spec (numpy array): mel-scaled spectrogram
            onsets (numpy array): list of onsets in units of frames.
            isbeat (numpy array): which onsets are beats (list of 0/1)
            beats (numpy array): list of beats (ground truth) in units seconds.
        """
        spec  = self.get_spec()
        onsets, isbeat = self.get_onsets_and_isbeat()
        beats = self.get_beats()
        return spec, onsets, isbeat, beats

    def augment(self, stretch_low=2/3, stretch_high=4/3):
        r"""Changes the offset and the stretching of `self` to generate a new
        sample. This is usually called for each sample of a whole dataset to do
        data augmentation.
        """

        self.stretch = stretch_low + np.random.rand() * (stretch_high - stretch_low)
        self.song_duration *= self.stretch
        self.offset  = np.random.rand() * (self.song_duration - self.duration)

    def correct(self, tightness=500):
        r"""Recompute which onsets are beats by first alligning the ground truth
        on the onsets. This can be useful if there are some misplaced beats.
        """

        spec, onsets, isbeat, beats = self.get_data()
        corrected_beats  = utils.correct_beats(onsets, beats)
        onsets, isbeat = utils.onsets_and_isbeat(spec, corrected_beats)
        utils.save_onsets_and_isbeat(self.onsets_file, onsets, isbeat)

    def predicted_beats(self, tightness=300):
        r"""Returns the list of beats (in seconds) and the bpm computed from the
        onsets (`onsets`) and the subset of those that are beats (`isbeat`).
        This uses a dynamical programming algorithm to compute the beat track on
        the list of onsets that are beats. Once a `BeatFinder` model has been
        trained, it can predict `isbeat` from `spec` and `onsets`, and this
        method spits out the actual beat track.
        """

        onsets, isbeat = self.get_onsets_and_isbeat()
        beats_frames = onsets[isbeat == 1]

        # If no beats or only one has been detected, we don't have to run the
        # beat track algorithm.
        if len(beats_frames) == 0 or len(beats_frames) == 1:
            return librosa.frames_to_time(beats_frames, constants.sr, constants.hl), np.nan

        # This is the dynamical programming part.
        beats, bpm = utils.beat_track(onsets[isbeat == 1], tightness)

        return beats, bpm

class AudioBeatsDataset(Dataset):
    r"""This is the basic dataset class to train `model.BeatFinder`. Each item
    is an `AudioBeats` object.

    Arguments:
        audiobeats_list (list): A list of AudioBeats objects. An
            `AudioBeatsDataset` object can be instantiated with either such a
            list or with a presaved `file` (see below).
        transform (class): A callable class which takes an `AudioBeats` object
            and returns something. In this package, it is always used with
            `model.ToTensor`.
    """

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

    def __add__(self, other):
        return ConcatAudioBeatsDataset([self, other])

    def precompute(self, mode='all', full=False):
        r"""Precomputes all the `AudioBeats` objects. This can take a
        substantial amount of time.

        Arguments:
            mode (str): Can be one of the following three choices:
                        'all': precomputes all data
                        'spec': precomputes only the spectrograms
                        'onsets_and_isbeat': precomputes only the onsets and
                            whether they are beats.
        """
        for j, audiobeats in enumerate(self):
            t = time.time()
            if mode == 'all':
                audiobeats.precompute()
            elif mode == 'spec':
                audiobeats.precompute_spec()
            elif mode == 'onsets_and_isbeat':
                audiobeats.precompute_onsets_and_isbeat(full)
            else:
                raise ValueError('Unknown mode')
            t = time.time() - t
            eta = str(datetime.timedelta(seconds=int(t * (len(self) - j - 1))))

            # Verbose loading.
            print(f' {100*(j+1)/len(self):6.2f}% | ETA: {eta} | {audiobeats.name}' + 20 * ' ', end='\r')

    def save(self, file):
        r"""Save the dataset in a file. This is saved as a csv-style file where
        each row stores the information of an `AudioBeats` object (recall that
        those do not contain any actual data, but only link to portions of some
        files.)

        Arguments:
            file (str): The relative path of the file where to save the dataset.
                If the enclosing directory doesn't exist, it will be created.
        """

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

    def augment(self, stretch_low=2/3, stretch_high=4/3):
        r"""This if for data augmentation. It calls `augment` to each item so
        that the new dataset points to randomly stretched and offsetted samples.
        """

        for audiobeats in self.audiobeats_list:
            audiobeats.augment(stretch_low, stretch_high)

    def correct(self, tightness=500):
        r"""Calls the `correct` method on each item. This corrects for misplaced
        beats in the ground truth.
        """

        for i, audiobeats in enumerate(self.audiobeats_list):
            audiobeats.correct(tightness)
            print(f'{i+1}/{len(self)}', end='\r')

    def clean(self, d=0.08, tightness=300):
        r"""This removes certain pathological items from `self`. For each item,
        it computes the beat track from the onsets that are beats, and reject
        the item if this beat track has an F-measure <= 0.9 compared to the
        ground truth. For example, this can happen if almost no onset have been
        detected, or if the ground truth is not precise enough and falls too
        much ahead or behind onsets (rather than on it).
        """
        new_list = []
        for i, audiobeats in enumerate(self.audiobeats_list):
            ground_truth    = audiobeats.get_beats()
            correction, bpm = audiobeats.predicted_beats(tightness=tightness)
            F = utils.F_measure(ground_truth, correction, d=d)
            if (F != None) and F > 0.9:
                new_list.append(audiobeats)
            print(f'{i+1}/{len(self)}', end='\r')
        self.audiobeats_list = new_list


def load(file):
    r"""Return the `AudioBeatsDataset` saved in `file`.
    
    Argument:
        file (str): The relative path of a saved `AudioBeatsDataset` object,
            saved via the method `self.save`. An `AudioBeatsDataset` object can
            be instantiated with either such a file or with a list of
            `AudioBeats` objects (see above).
    Returns:
        dataset (AudioBeatsDataset): The dataset saved in `file`.
    """
    
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
        
    dataset = AudioBeatsDataset(audiobeats_list)
    
    return dataset
        
        
class SubAudioBeatsDataset(AudioBeatsDataset):
    r"""Subset of an `AudioBeatsDataset` at specified indices.

    Arguments:
        dataset (AudioBeatsDataset): The original `AudioBeatsDataset`.
        indices (list): Selected indices in the original `AudioBeatsDataset`.
    """

    def __init__(self, dataset, indices):
        audiobeats_list = [dataset.audiobeats_list[i] for i in indices]
        super().__init__(audiobeats_list, dataset.transform)


class AudioBeatsDatasetFromSong(AudioBeatsDataset):
    r"""An `AudioBeatsDataset` consisting of equally spaced `AudioBeats` of the
    same duration and completely covering a given audio file.

    Arguments:
        audio_file (str): The path of the audio file.
        beats_file (str): The path of the file containing all the beats (in
        seconds) in the audio file (a `.txt` file with one column of floats).
        precomputation_path (str): The directory where the data pointed by the
            `AudioBeats` objects is stored.
        transform (object): Transform of the `AudioBeatsDataset`.
        duration (float): The duration of the audio pointed by each
            `AudioBeats`.
        stretch (float): The amount by which to stretch the audio.
        force_nb_samples (int or None): By default (if `None`) there will be as
            many `AudioBeats` as possible, side-by-side starting from time
            zero, until no `AudioBeats` can fit in the full audio. Hence, there
            might be a small portion (of length < duration) at the end not
            covered by any `AudioBeats`. If `force_nb_samples` is set to a
            larger number, there will be more `AudioBeats` with zero padding.
            This is useful when we have a large list of, e.g., ~30 seconds
            audio files and want to split each of them in three `AudioBeats`.
            We set `force_nb_samples = 3` so that even if an audio file is
            slightly less or slightly more than 30 seconds we always get 3
            samples (possibly with some zero padding on the right of the last
            one).
    """

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
    r"""Concatenate multiple `AudioBeatsDataset`s.

    Arguments:
        datasets (list): A list of `AudioBeatsDataset` objects.
    """

    def __init__(self, datasets):
        for i in range(1, len(datasets)):
            if datasets[i].transform != datasets[i - 1].transform:
                raise ValueError("All transforms must be the same")

        audiobeats_list = []
        for dataset in datasets:
            audiobeats_list += dataset.audiobeats_list

        super().__init__(audiobeats_list, datasets[0].transform)


class AudioBeatsDatasetFromList(ConcatAudioBeatsDataset):
    r"""An `AudioBeatsDataset` instantiated from a file containing a list of
    audio files.

    Arguments:
        audio_files (str): The path of a `.txt` file where each line is the
            relative path of an audio file (relative to where `audio_files` is).
        precomputation_path (str): Where to store the precomputated data of each
            item.
        transform (object): transform of the `AudioBeatsDataset`.
        duration (float): Duration of each audio sample.
        stretch (float): The amount by which to stretch the audio files.
        force_nb_samples (int or None): To pass to `AudioBeatsDatasetFromSong`.
    """

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
