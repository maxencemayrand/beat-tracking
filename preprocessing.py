import librosa
import numpy as np
import pandas as pd
from constants import *


def onset_strength(spec=None, wav=None):
    onset_env = librosa.onset.onset_strength(y=wav, sr=sr, S=spec, hop_length=hl)
    onset_env[-1] = 0
    onset_env[:-1] = onset_env[1:]
    return onset_env

def get_spec(wav):
    spec = librosa.feature.melspectrogram(
            wav,
            sr=sr,
            hop_length=hl,
            fmin=fm,
            n_mels=nb,
            htk=htk
            ).astype(np.float32)
    spec = librosa.power_to_db(spec)
    return spec

def peak_detect(onset_env):
    return librosa.util.peak_pick(onset_env,
                                  pre_max=pre_max, 
                                  post_max=post_max,
                                  pre_avg=pre_avg,
                                  post_avg=post_avg,
                                  wait=wait,
                                  delta=delta)

def get_onsets(spec, beats_times):
    """Old version. I'm now always using `get_onsets_from_beats` instead. """
    onset_envelope = librosa.onset.onset_strength(S=spec)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_envelope,
                                        pre_max=pre_max, 
                                        post_max=post_max,
                                        pre_avg=pre_avg,
                                        post_avg=post_avg,
                                        wait=wait,
                                        delta=delta)
    onsets_times = librosa.frames_to_time(onsets, sr, hl)
    dist = np.min(np.abs(onsets_times[:, np.newaxis] - beats_times), axis=1)
    onsets_selected = onsets[dist < delta]
    df = pd.DataFrame()
    df['onsets'] = onsets
    df['isbeat'] = (dist < delta).astype(np.int)
    return df

def select_onsets(onsets_times, beats_times, dt=dt):
    closests_idx = np.argmin(np.abs(beats_times[:, np.newaxis] - onsets_times), axis=1)
    closests = onsets_times[closests_idx]
    selected_idx = closests_idx[np.abs(closests - beats_times) < dt]
    
    return selected_idx

def get_onsets_from_beats(spec, beats, dt=dt):
    onset_envelope = onset_strength(spec=spec)
    onsets = librosa.onset.onset_detect(
                onset_envelope=onset_envelope, 
                pre_max=pre_max, 
                post_max=post_max,
                delta=delta)
    onsets_times = librosa.frames_to_time(onsets, sr, hl)
    
    selected_idx = select_onsets(onsets_times, beats, dt)
    
    isbeat = np.zeros_like(onsets)
    isbeat[selected_idx] = 1
    
    return onsets, isbeat

def save_onsets(file, onsets, isbeat):
        df = pd.DataFrame()
        df['onsets'] = onsets
        df['isbeat'] = isbeat
        df.to_csv(file, index=False)

def correct_isbeat(onsets, corrected_beats):
    onsets_times = librosa.frames_to_time(onsets, sr, hl)
    selected_idx = select_onsets(onsets_times, corrected_beats)
    corrected_isbeat = np.zeros(len(onsets), dtype=onsets.dtype)
    corrected_isbeat[selected_idx] = 1
    return corrected_isbeat

