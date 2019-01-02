"""Various utilities to process audio"""

import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import constants

def onset_strength(spec=None, wav=None):
    r"""Returns the onset envelope of either a spectrogram or a wav. Computed
    with librosa with the constants specified in `beatfinder.constants`.

    Arguments:
        spec (2d numpy array): A spectrogram.
        wav (1d numpy array): An audio wav.

    Returns:
        onset_env (1d numpy array): The onset envelope.
    """
    onset_env = librosa.onset.onset_strength(y=wav, sr=constants.sr, S=spec, hop_length=constants.hl)
    onset_env -= onset_env.min()
    onset_env /= onset_env.max()
    onset_env[:-constants.offset] = onset_env[constants.offset:]
    onset_env[-constants.offset:] = 0
    return onset_env

def spectrogram(wav):
    r"""Returns the log power mel-scaled spectrogram of an audio wav computed
    with librosa with the constants specified in `beatfinder.constants`.

    Argument:
        wav (1d numpy array): The audio wav.

    Returns:
        spec (2d numpy array): The log power mel-scaled spectrogram of `wav`.
    """

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
    r"""Returns the peaks of an onset envelope with the constants specified in
    `beatfinder.constants`.

    Argument:
        onset_env (1d numpy array): An onset envelope.

    Returns:
        onsets (1d numpy array): The indices of `onset_env` where a peak has
            been detected.
    """

    onsets = librosa.util.peak_pick(onset_env,
                                  pre_max=constants.pre_max,
                                  post_max=constants.post_max,
                                  pre_avg=constants.pre_avg,
                                  post_avg=constants.post_avg,
                                  wait=constants.wait,
                                  delta=constants.delta)
    return onsets

def select_onsets(onsets_times, beats_times, dt=constants.dt):
    r"""Select which onsets are beats according to whether they are within a
    small interval of a beat.

    Arguments:
        onsets_times (1d numpy array): Onsets in units of seconds.
        beats_times (1d numpy array): Beats (ground truth) in units of seconds.
        dt (float): The maximum distance from a beat that an onset can be to be
            selected as a beat.

    Returns:
        selected_idx (1d numpy array): The indices of `onsets_times` which give
            onsets that are within `dt` of any element of `beats_times`.
    """

    closests_idx = np.argmin(np.abs(beats_times[:, np.newaxis] - onsets_times), axis=1)
    closests = onsets_times[closests_idx]
    selected_idx = closests_idx[np.abs(closests - beats_times) < dt]

    return selected_idx


def onsets_and_isbeat(spec, beats, dt=constants.dt):
    r"""Returns the onsets and the array specifying which ones are selected as
    beats from a spectrogram and a list of beats.

    Arguments:
        spec (2d numpy array): A spectrogram.
        beats (1d numpy array): Beats (ground truth) in units of seconds.
        dt (float): To be passed to `select_onsets`.

    Returns:
        onsets (1d numpy array): The onsets in units of frames.
        isbeat (1d numpy array): `isbeat[i] = 1` if `onsets[i]` has been
            selected as beat and `0` otherwise.
    """

    onset_env = onset_strength(spec=spec)
    onsets = peak_detect(onset_env)
    onsets_times = librosa.frames_to_time(onsets, constants.sr, constants.hl)

    selected_idx = select_onsets(onsets_times, beats, dt)

    isbeat = np.zeros_like(onsets)
    isbeat[selected_idx] = 1

    return onsets, isbeat

def save_onsets_and_isbeat(file, onsets, isbeat):
    r"""Save the `onsets` and `isbeat` data in a file.

    Arguments:
        file (str): Path of the file to write the data.
        onsets (1d numpy array): The onsets in units of frames.
        isbeat (1d numpy array): `isbeat[i] = 1` if `onsets[i]` has been
            selected as beat and `0` otherwise.
    """
    df = pd.DataFrame()
    df['onsets'] = onsets
    df['isbeat'] = isbeat
    df.to_csv(file, index=False)

def bpm_estimation(beats_frames, max_dev=constants.dt, show_plot=True):
    r"""Estimates the BPM (beats per minutes) from a possibly incomplete and
    inacurate list of beats (e.g. as predicted by a `BeatFinder` model).

    The algorithm computes all the time differences between successive beats
    and hence obtain a list of possible BPMs. It then takes the median and
    computes the mean of all the BPMs that are close enough to this median.

    Arguments:
        beats_frames (1d numpy array): The beats in units of frames.
        max_dev (float): The maximum distance from the median BPM to be
            included in the computation of the mean.
        show_plot (bool): Show the plot of all the BPMs and an horizontal line
            on the final BPM computed.

    Returns:
        bpm (float): The BPM computed.
    """
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
    r"""Returns the BPM (beats per minutes) of a list of beats, assuming the
    list is complete and accurate.

    Arguments:
        beats (1d numpy array): The beats in units of seconds.

    Returns:
        bpm (float): The BPM of `beats`.
    """

    bpm = 60 / (beats[1:] - beats[:-1]).mean()

    return bpm

def beat_track(onsets, tightness=300, bpm=None):
    r"""Returns the beat track corresponding to a list of onsets. This is computed using the dynamic programming algorithm discovered by [1] and implemented by librosa (10.5281/zenodo.1342708).

    We only use the core method `librosa.beat.__beat_track_dp` from librosa rather than their standard method `librosa.beat.beat_track` since in our specific case we can simplify some of the preprocessing done by `librosa.beat.beat_track` to get faster and more accurate results. In particular, we modified the

    [1] Ellis, Daniel PW. "Beat tracking by dynamic programming."
        Journal of New Music Research 36.1 (2007): 51-60.
        http://labrosa.ee.columbia.edu/projects/beattrack/

    Arguments:
        onsets (1d numpy array): The onsets in units of frames.
        tightness (float): The tightness of the dynamic programming algorithm.
        bpm (None or float): If provided, the beat track will try to be of that
            BPM. Otherwise, the BPM is first estimated using `bpm_estimation`.

    Returns:
        beats_times (1d numpy array): The beats computed in units of seconds.
        bpm (float): The BPM estimated.
    """

    if not bpm:
        bpm = bpm_estimation(onsets, show_plot=False)

    max_frame = onsets.max() + 10
    fft_res = constants.sr / constants.hl
    period = round(60.0 * fft_res / bpm)
    localscore = np.zeros(max_frame, dtype=np.float)
    localscore[onsets] = 10

    # This is the core dynamic programming from [1] and implemented by librosa.
    backlink, cumscore = librosa.beat.__beat_track_dp(localscore, period, tightness)

    # Get the local maximum of the cumulative scores.
    maxes = librosa.util.localmax(cumscore)
    # The next line is important as sometimes the above method falsely detect a
    # local maximum at the very last frame and the beat track then gets
    # contracted to match it.
    maxes[-1] = False

    med_score = np.median(cumscore[np.argwhere(maxes)])
    last_beat = np.argwhere((cumscore * maxes * 2 > med_score)).max()

    # Reconstruct the beat path from backlinks
    beats = [last_beat]
    while backlink[beats[-1]] >= 0:
        beats.append(backlink[beats[-1]])
    beats = np.array(beats[::-1], dtype=int)

    # Convert frames to times
    beats_times = librosa.frames_to_time(beats, constants.sr, constants.hl)

    return beats_times, bpm

def correct_beats(onsets, beats, tightness=500):
    r"""Realign beats to fit better on the list of onsets.

    Arguments:
        onsets (1d numpy array): Onsets in units of frames.
        beats (1d numpy array): Beats in units of seconds.
        tightness (float): The tightness of the dynamic programming beat track
            algorithm.

    Returns:
        corrected_beats (1d numpy array): The realigned beats in units of
            seconds.
    """

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
    r"""Recompute which onsets are selected as beats from the corrected beats.

    Arguments:
        onsets (1d numpy array): Onsets in units of frames.
        corrected_beats (1d numpy array): The realigned beats in units of
            seconds.

    Returns:
        corrected_isbeat (1d numpy array): `corrected_isbeat[i] = 1` if
            onsets[i] is selected as a beat and `0` otherwise.
    """

    onsets_times = librosa.frames_to_time(onsets, constants.sr, constants.hl)
    selected_idx = select_onsets(onsets_times, corrected_beats)
    corrected_isbeat = np.zeros(len(onsets), dtype=onsets.dtype)
    corrected_isbeat[selected_idx] = 1
    return corrected_isbeat

def measures(tn, fp, fn, tp):
    r"""Returns the accuracy, precision, recall, and F-measure.

    Arguments:
        tn (int): Number of true positives.
        fp (int): Number of false positives.
        fn (int): Number of false negatives.
        tp (int): Number of true positives.

    Returns:
        a (float): Accuracy.
        p (float): Precision.
        r (float): Recall.
        F (float): F-measure.
    """

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
    r"""Returns the F-measure of a beat track prediction. Two beats are
    considered the same if they are less than `d` seconds from each other.

    Arguments:
        ground_truth (1d numpy array): The ground truth beats in units of
            seconds.
        prediction (1d numpy array): The predicted beats in units of seconds.
        d (float): The maximum distance between two beats to be considered the
            same.

    Returns:
        F (float or None): The F-measure, if well-defined.
    """

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

def predict_isbeat(model, audiobeats, totensor):
    r"""Predict the `isbeat` data (the onsets which are beats) from a
    `BeatFinder` model.

    Arguments:
        model (BeatFinder): Model used to make the prediction.
        audiobeats (AudioBeats): The audio sample to make the prediction.
        totensor (ToTensor): The transform from AudioBeats to pytorch tensors.

    Returns:
        isbeat_pred (1d numpy): The prediction.
    """

    spec_tensor, onsets_tensor, _ = totensor(audiobeats)
    spec_tensor = spec_tensor.unsqueeze(0)
    onsets_tensor = onsets_tensor.unsqueeze(0)
    onsets, _ = audiobeats.get_onsets_and_isbeat()
    isbeat_pred = model.predict(spec_tensor, onsets_tensor).numpy().flatten()[onsets]
    return isbeat_pred

def predict_beats(model, audiobeats, totensor, tightness=300):
    r"""Predict the beats of an audio sample from a `BeatFinder` model.

    Arguments:
        model (BeatFinder): Model used to make the prediction.
        audiobeats (AudioBeats): The audio sample to make the prediction.
        totensor (ToTensor): The transform from AudioBeats to pytorch tensors.
        tightness (float): To pass to the dynamic programming beat tracking
            algorithm.

    Returns:
        beats_frames (1d numpy array): The onsets selected as beats in units of
            frames.
        predicted_beats (1d numpy array): The beats in units of seconds.
        bpm (float): The predicted BPM.
    """
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
    r"""Returns the F-measure of a whole `AudioBeatsDataset`.

    Arguments:
        model (BeatFinder): Model used to make the predictions.
        dataset (AudioBeatsDataset): The dataset to evaluate.
        totensor (ToTensor): The transform from AudioBeats to pytorch tensors.

    Returns:
        F (float): The mean of the F-measures for each sample.
        F_nan (int): The number of samples for which the F-measure was not
            defined.
    """
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

    F = F_sum / (len(dataset) - F_nan)
    return F, F_nan

def F_measure_with_librosa_comparison(model, dataset, totensor):
    r"""Same as `F_measure_from_dataset` but also compares the result with
    librosa's prediction.
    """
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


def tp_fn_fp(ground_truth, predicted_beats, d=0.07):
    r"""Returns the true positives, false negatives, and false positives of a list of beats.

    Arguments:
        ground_truth (1d numpy array): The ground truth beats in seconds.
        predicted_beats (1d numpy array): The predicted beats in seconds.
        d (float): The maximum distance between two beats to be considered the same.

    Returns:
        tp (int): True positives.
        fn (int): False negatives.
        fp (int): False positives.
    """
    if len(ground_truth) == 0 or len(predicted_beats) == 0:
        tp = 0
        fn = len(ground_truth)
        fp = len(predicted_beats)
    else:
        dists = np.abs(predicted_beats[:, np.newaxis] - ground_truth)
        tp = np.sum(np.min(dists, axis=1) < d)
        fn = np.sum(np.min(dists, axis=0) >= d)
        fp = np.sum(np.min(dists, axis=1) >= d)
    
    return tp, fn, fp

def F(tp, fn, fp):
    if tp + fp == 0 or tp + fn == 0 or tp == 0:
        F = np.nan
    else:
        p = tp / (tp + fp)      # precision
        r = tp / (tp + fn)      # recall
        F = 2 * p * r / (p + r) # F-measure
    return F

def evaluate_dataset(dataset, model, totensor=None, d=0.07, verbose=False):
    r"""Returns the true positives, false negatives, and false positives an `AudioBeatsDataset`.

    Arguments:
        model (BeatFinder): Model used to make the predictions.
        dataset (AudioBeatsDataset): The dataset to evaluate.
        totensor (ToTensor): The transform from AudioBeats to pytorch tensors.

    Returns:
        total_tp (int): True positives.
        total_fn (int): False negatives.
        total_fp (int): False positives.
    """
    total_tp = 0
    total_fn = 0
    total_fp = 0
    for i, audiobeats in enumerate(dataset):
        if model=='librosa':
            spec = audiobeats.get_spec()
            onset_env = librosa.onset.onset_strength(S=spec)
            bpm, beats_frames = librosa.beat.beat_track(onset_envelope=onset_env)
            predicted_beats = librosa.frames_to_time(beats_frames, constants.sr, constants.hl)
        else:
            beats_frames, predicted_beats, bpm = predict_beats(model, audiobeats, totensor)
        
        ground_truth = audiobeats.get_beats()
        
        tp, fn, fp = tp_fn_fp(ground_truth, predicted_beats, d)
        
        total_tp += tp
        total_fn += fn
        total_fp += fp
        
        if verbose:
            print(f'{i+1:3d}/{len(dataset)} | tp: {tp:3d} | fn: {fn:3d} | fp: {fp:3d}')

    return total_tp, total_fn, total_fp



















