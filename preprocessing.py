import librosa
import numpy as np
import pandas as pd
from constants import *

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

def get_onsets(spec, beats_times):
    onset_envelope = librosa.onset.onset_strength(S=spec)
    onsets = librosa.onset.onset_detect(
                onset_envelope=onset_envelope, 
                pre_max=pre_max, 
                post_max=post_max,
                delta=delta)
    onsets_times = librosa.frames_to_time(onsets, sr, hl)
    dist = np.min(np.abs(onsets_times[:, np.newaxis] - beats_times), axis=1)
    onsets_selected = onsets[dist < delta]
    df = pd.DataFrame()
    df['onsets'] = onsets
    df['isbeat'] = (dist < delta).astype(np.int)
    return df

def get_onsets_from_beats(spec, beats, dt=dt):
    onset_envelope = librosa.onset.onset_strength(S=spec)
    onsets = librosa.onset.onset_detect(
                onset_envelope=onset_envelope, 
                pre_max=pre_max, 
                post_max=post_max,
                delta=delta)
    onsets_times = librosa.frames_to_time(onsets, sr, hl)
    
    closests_idx = np.argmin(np.abs(beats[:, np.newaxis] - onsets_times), axis=1)
    closests = onsets_times[closests_idx]
    selected_idx = closests_idx[np.abs(closests - beats) < dt]
    
    isbeat = np.zeros_like(onsets)
    isbeat[selected_idx] = 1
    
    return onsets, isbeat
    
def save_onsets(file, onsets, isbeat):
        df = pd.DataFrame()
        df['onsets'] = onsets
        df['isbeat'] = isbeat
        df.to_csv(file, index=False)

    
    
    
    