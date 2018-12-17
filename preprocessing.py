import librosa
import numpy as np
import pandas as pd
from constants import *

def get_spec(file):
    wav = librosa.load(file, sr)[0]
    spec = librosa.feature.melspectrogram(
            wav,
            sr=sr,
            hop_length=hl,
            fmin=fm,
            n_mels=nb,
            htk=True
            ).astype(np.float32)
    spec = spec.T
    spec = librosa.power_to_db(spec)
    return spec

def get_onsets(spec, beats_times):
    onset_envelope = librosa.onset.onset_strength(S=spec)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_envelope)
    onsets_times = librosa.frames_to_time(onsets, sr, hl)
    dist = np.min(np.abs(onsets_times[:, np.newaxis] - beats_times), axis=1)
    onsets_selected = onsets[dist < delta]
    df = pd.DataFrame()
    df['onsets'] = onsets
    df['isbeat'] = (dist < delta).astype(np.int)
    return df