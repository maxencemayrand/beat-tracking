import numpy as np
from constants import *
import librosa
import matplotlib.pyplot as plt
import preprocessing

def bpm_estimation(onset_env, max_dev=dt, show_plot=True):
    times = librosa.frames_to_time(np.argwhere(onset_env == 1).flatten(), sr, hl)
    diffs = times[1:] - times[:-1]
    median_diff = np.median(diffs) # s/b
    selected_diff = diffs[np.abs(diffs - median_diff) < max_dev]
    if len(selected_diff) > 0:
        mean_diff = np.mean(selected_diff)
    else:
        print("mean didn't work")
        mean_diff = median_diff
    bpm = 60 / mean_diff
    if show_plot == True:
        print(bpm)
        plt.plot(diffs)
        plt.plot(np.arange(len(diffs)), np.zeros(len(diffs)) + mean_diff);
    return bpm

def ground_truth_bpm(beats):
    return 60 / (beats[1:] - beats[:-1]).mean()

def beat_tracker(beats_frames, max_frame, bpm, tightness):
    fft_res = sr / hl
    period = round(60.0 * fft_res / bpm)
    localscore = np.zeros(max_frame, dtype=np.float)
    localscore[beats_frames] = 10
    backlink, cumscore = librosa.beat.__beat_track_dp(localscore, period, tightness)

    # get the position of the last beat
    maxes = librosa.util.localmax(cumscore)
    maxes[-1] = False # This is important.
    med_score = np.median(cumscore[np.argwhere(maxes)])
    last_beat = np.argwhere((cumscore * maxes * 2 > med_score)).max()

    # Reconstruct the beat path from backlinks
    beats = [last_beat]
    while backlink[beats[-1]] >= 0:
        beats.append(backlink[beats[-1]])

    beats = np.array(beats[::-1], dtype=int)

    return beats

def correct_beats(onsets, beats):
    bpm = ground_truth_bpm(beats)
    onsets_times = librosa.frames_to_time(onsets, sr, hl)
    selected_idxs = preprocessing.select_onsets(onsets_times, beats)
    selected_onsets = onsets[selected_idxs]
    selected_onsets_times = onsets_times[selected_idxs]
    
    corrected_beats = beat_tracker(selected_onsets, onsets.max() + 10, bpm, 500)
    corrected_beats_times = librosa.frames_to_time(corrected_beats, sr, hl)
    
    return corrected_beats_times








