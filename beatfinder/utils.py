import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import constants

def onset_strength(spec=None, wav=None):
    onset_env = librosa.onset.onset_strength(y=wav, sr=constants.sr, S=spec, hop_length=constants.hl)
    onset_env -= onset_env.min()
    onset_env /= onset_env.max()
    onset_env[:-constants.offset] = onset_env[constants.offset:]
    onset_env[-constants.offset:] = 0
    return onset_env

def spectrogram(wav):
    spec = librosa.feature.melspectrogram(
            wav,
            sr=constants.sr,
            hop_length=constants.hl,
            fmin=constants.fm,
            n_mels=constants.nb,
            htk=constants.htk
            ).astype(np.float32)
    spec = librosa.power_to_db(spec)
    return spec

def peak_detect(onset_env):
    return librosa.util.peak_pick(onset_env,
                                  pre_max=constants.pre_max, 
                                  post_max=constants.post_max,
                                  pre_avg=constants.pre_avg,
                                  post_avg=constants.post_avg,
                                  wait=constants.wait,
                                  delta=constants.delta)

def select_onsets(onsets_times, beats_times, dt=constants.dt):
    closests_idx = np.argmin(np.abs(beats_times[:, np.newaxis] - onsets_times), axis=1)
    closests = onsets_times[closests_idx]
    selected_idx = closests_idx[np.abs(closests - beats_times) < dt]
    
    return selected_idx


def onsets_and_isbeat(spec, beats, dt=constants.dt):
    onset_env = onset_strength(spec=spec)
    onsets = peak_detect(onset_env)
    onsets_times = librosa.frames_to_time(onsets, constants.sr, constants.hl)
    
    selected_idx = select_onsets(onsets_times, beats, dt)
    
    isbeat = np.zeros_like(onsets)
    isbeat[selected_idx] = 1
    
    return onsets, isbeat

def save_onsets_and_isbeat(file, onsets, isbeat):
        df = pd.DataFrame()
        df['onsets'] = onsets
        df['isbeat'] = isbeat
        df.to_csv(file, index=False)

def bpm_estimation(beats_frames, max_dev=constants.dt, show_plot=True):
    beats_times = librosa.frames_to_time(beats_frames, constants.sr, constants.hl)
    bpms = 60 / (beats_times[1:] - beats_times[:-1])
    median_bpm = np.median(bpms) # s/b
    selected_bpms = bpms[np.abs(bpms - median_bpm) < max_dev]
    if len(selected_bpms) > 0:
        bpm = np.mean(selected_bpms)
    else:
        bpm = median_bpm
    if show_plot == True:
        print(bpm)
        plt.plot(bpms)
        plt.plot(np.arange(len(bpms)), np.zeros(len(bpms)) + bpm);
    return bpm

def ground_truth_bpm(beats):
    return 60 / (beats[1:] - beats[:-1]).mean()

def beat_track(onsets, tightness=300, bpm=None):
    if not bpm:
        bpm = bpm_estimation(onsets, show_plot=False)
    max_frame = onsets.max() + 10
    fft_res = constants.sr / constants.hl
    period = round(60.0 * fft_res / bpm)
    localscore = np.zeros(max_frame, dtype=np.float)
    localscore[onsets] = 10
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
    
    beats_times = librosa.frames_to_time(beats, constants.sr, constants.hl)
    
    return beats_times, bpm

def correct_beats(onsets, beats, tightness=500):
    if len(beats) == 0 or len(beats) == 1:
        return beats
    bpm = ground_truth_bpm(beats)
    onsets_times = librosa.frames_to_time(onsets, constants.sr, constants.hl)
    selected_idxs = select_onsets(onsets_times, beats)
    
    if len(selected_idxs) == 0:
        return np.array([], dtype=beats.dtype)
    if len(selected_idxs) == 1:
        return onsets_times[selected_idxs]
    
    selected_onsets = onsets[selected_idxs]
    corrected_beats = beat_track(selected_onsets, tightness=tightness, bpm=bpm)[0]
    
    return corrected_beats

def correct_isbeat(onsets, corrected_beats):
    onsets_times = librosa.frames_to_time(onsets, constants.sr, constants.hl)
    selected_idx = select_onsets(onsets_times, corrected_beats)
    corrected_isbeat = np.zeros(len(onsets), dtype=onsets.dtype)
    corrected_isbeat[selected_idx] = 1
    return corrected_isbeat

def measures(tn, fp, fn, tp):
    a = (tp + tn) / (tn + fp + fn + tp) # accuracy
    if tp + fp != 0 and tp + fn != 0:
        p = tp / (tp + fp)      # precision
        r = tp / (tp + fn)      # recall
        if tp != 0:
            F = 2 * p * r / (p + r) # F-measure
        else:
            F = np.nan
    else:
        p = np.nan
        r = np.nan
        F = np.nan
    return a, p, r, F

def F_measure(ground_truth, prediction, d=0.07):
    if len(ground_truth) == 0:
        return None
    if len(prediction) == 0:
        return None
    
    dists = np.abs(prediction[:, np.newaxis] - ground_truth)
    tp = np.sum(np.min(dists, axis=1) < d)
    fn = np.sum(np.min(dists, axis=0) >= d)
    fp = np.sum(np.min(dists, axis=1) >= d)
    
    if tp + fp == 0 or tp + fn == 0 or tp == 0:
        return None
    else:
        p = tp / (tp + fp)      # precision
        r = tp / (tp + fn)      # recall
        F = 2 * p * r / (p + r) # F-measure
        return F

def onsets_selected(onsets, isbeat):
    return onsets[isbeat == 1]
    
def onsets_selected_times(onsets, isbeat):
    return librosa.frames_to_time(onsets_selected(onsets, isbeat), constants.sr, constants.hl)
    
def predict_isbeat(model, audiobeats, totensor):
    spec_tensor, onsets_tensor, _ = totensor(audiobeats)
    spec_tensor = spec_tensor.unsqueeze(0)
    onsets_tensor = onsets_tensor.unsqueeze(0)
    onsets, _ = audiobeats.get_onsets_and_isbeat()
    isbeat_pred = model.predict(spec_tensor, onsets_tensor).numpy().flatten()[onsets]
    return isbeat_pred

def predict_beats(model, audiobeats, totensor, tightness=300):
    spec_tensor, onsets_tensor, _ = totensor(audiobeats)
    spec_tensor = spec_tensor.unsqueeze(0)
    onsets_tensor = onsets_tensor.unsqueeze(0)
    onsets, _ = audiobeats.get_onsets_and_isbeat()
    isbeat_pred = model.predict(spec_tensor, onsets_tensor).cpu().numpy().flatten()[onsets]
    beats_frames = onsets[isbeat_pred == 1]
    if len(beats_frames) == 0:
        predicted_beats = np.array([], dtype=np.float32)
        bpm = None
    elif len(beats_frames) == 1:
        predicted_beats = beats_frames
        bpm = None
    else:
        predicted_beats, bpm = beat_track(beats_frames, tightness)
    
    return beats_frames, predicted_beats, bpm
    
def F_measure_from_dataset(model, dataset, totensor):
    F_sum = 0
    F_nan = 0
    for i, audiobeats in enumerate(dataset):
        beats_frames, predicted_beats, bpm = predict_beats(model, audiobeats, totensor)
        beats = audiobeats.get_beats()
        F = F_measure(beats, predicted_beats)
        if F == None:
            print(f'{i:3d} :  NaN | ', end='')
            F_nan += 1
        else:
            print(f'{i:3d} : {F:.2f} | ', end='')
            F_sum += F
        if (i + 1) % 8 == 0:
            print()

    return F_sum / (len(dataset) - F_nan), F_nan

def F_measure_with_librosa_comparison(model, dataset, totensor):
    F_model_sum = 0
    F_model_nan = 0
    F_libro_sum = 0
    F_libro_nan = 0
    for i, audiobeats in enumerate(dataset):
        beats_frames, predicted_beats, bpm = predict_beats(model, audiobeats, totensor)
        beats = audiobeats.get_beats()
        F_model = F_measure(beats, predicted_beats)
        if F_model == None:
            print(f'{i:3d} |   NaN ', end='')
            F_model_nan += 1
        else:
            print(f'{i:3d} | {F_model:.3f} ', end='')
            F_model_sum += F_model
        
        spec = audiobeats.get_spec()
        onset_env = librosa.onset.onset_strength(S=spec)
        bpm, beats_frames = librosa.beat.beat_track(onset_envelope=onset_env)
        librosa_beats = librosa.frames_to_time(beats_frames, constants.sr, constants.hl)
        F_libro = F_measure(beats, librosa_beats)
        if F_libro == None:
            print(f'  NaN ')
            F_libro_nan += 1
        else:
            print(f' {F_libro:.3f}')
            F_libro_sum += F_libro
    
    F_model = F_model_sum / (len(dataset) - F_model_nan)
    F_libro = F_libro_sum / (len(dataset) - F_libro_nan)
    
    return F_model, F_model_nan, F_libro, F_libro_nan
