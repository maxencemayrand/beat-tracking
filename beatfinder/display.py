import matplotlib.pyplot as plt
import librosa
import numpy as np

from . import utils
from . import constants

def showdata(audiobeats, duration=None, offset=None, beatfinder=None, device=None, showpred=True):
    r"""Displays the data pointed by an AudioBeats object. It shows the onset envelope,
    the onsets, the onsets selected as beats, the ground truth beats, and the spectrogram.

    Arguments:
        audiobeats (AudioBeats): The `AudioBeats` object to display.
        duration (float): To see a smaller portion of the data, this can be set
            to a lower value than the duration of `audiobeats`.
        offset (float): Display only the data starting from `offset` (combine)
            with `duration` to see a smaller portion of the data).
        beatfinder (BeatFinder): A model, if available.
        device (torch.device): The device of the model.
        showpred (bool): To show the predicted beats.
    """

    spec, onsets, isbeat, beats = audiobeats.get_data()
    if duration == None:
        duration = audiobeats.duration
    if offset == None:
        offset = 0

    if showpred:
        pred_beats, _ = audiobeats.predicted_beats()

    onsets_times = librosa.frames_to_time(onsets, constants.sr, constants.hl)

    onsets_selected = onsets[isbeat == 1]
    onsets_selected_times = librosa.frames_to_time(onsets_selected, constants.sr, constants.hl)

    onset_envelope = utils.onset_strength(spec=spec)

    times = librosa.frames_to_time(np.arange(len(onset_envelope)), constants.sr, constants.hl)

    total_duration = librosa.frames_to_time(spec.shape[1], constants.sr, constants.hl)

    plt.figure(figsize=(16, 8))
    plt.subplots_adjust(hspace=0)

    plt.subplot(4, 1, 1)
    if audiobeats.beats_file:
        plt.vlines(beats, 2, 3, color='g', label='Ground truth\nbeats')
        plt.ylim(0, 3)
    else:
        plt.ylim(0, 2)
    if showpred:
        plt.vlines(pred_beats, 1, 2, color='b', label='Predicted beats')
        plt.vlines(onsets_selected_times, 0, 1,  color='m', linestyles='-', alpha=1, label='Onsets selected\nas beats')
    else:
        plt.vlines(onsets_selected_times, 1, 2,  color='m', linestyles='-', alpha=1, label='Onsets selected\nas beats')
        plt.ylim(1, 3)
    plt.xlim(offset, offset + duration)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.legend(frameon=True, framealpha=0.75, bbox_to_anchor=(1.15, 1));

    plt.subplot(4, 1, 2)
    plt.vlines(onsets_times, 0, 1, color='k', linestyles='--', alpha=0.3, label='Onsets')

    if beatfinder:
        probs = audiobeats.probabilities(beatfinder, device)
        plt.vlines(onsets_times, 0, probs[onsets],  color='r', linewidths=7, alpha=0.25,
                   label='Probability of the\nonset to be a beat')
    else:
        plt.vlines(onsets_selected_times, 0, 1,  color='m', linestyles='--', alpha=1, label='Onsets selected\nas beats')
    plt.plot(times, onset_envelope, label='Onset envelope')
    plt.xlim(offset, offset + duration)
    plt.ylim(0, 1)
    plt.xticks([], [])
    plt.legend(frameon=True, framealpha=0.75, bbox_to_anchor=(1.15, 1));

    plt.subplot(2, 1, 2)
    freq = librosa.mel_frequencies(n_mels=constants.nb, fmin=constants.fm, htk=constants.htk)
    plt.pcolormesh(times, freq, spec)
    plt.xlabel('Time [seconds]')
    plt.ylabel('Frequency [Hz]')
    plt.xlim(offset, offset + duration)

def showspec(spec):
    r"""Display a spectrogram.

    Arguments:
        spec (2d numpy array): The spectrogram to display.
    """
    plt.figure(figsize=(16, 4))
    times = librosa.frames_to_time(np.arange(spec.shape[1]), constants.sr, constants.hl)
    freq = librosa.mel_frequencies(n_mels=constants.nb, fmin=constants.fm, htk=constants.htk)
    plt.pcolormesh(times, freq, spec)

def showprediction(ground_truth, predicted_beats, onsets_selected):
    r"""Display the final beat track predicted together with the selected
    onsets which have been used to compute it and the ground truth for
    comparison.

    Arguments:
        ground_truth (1d numpy array): The ground truth beats in units of
            seconds.
        predicted_beats (1d numpy array): The beats predicted in units of
            seconds.
        onsets_selected (1d numpy array): The onsets selected as beats in units
            of frames.
    """
    onsets_selected_times = librosa.frames_to_time(onsets_selected, constants.sr, constants.hl)
    plt.figure(figsize=(16, 2))
    plt.vlines(ground_truth, 2, 3, color='g', label='Ground truth')
    plt.vlines(predicted_beats, 1, 2, color='b', label='Prediction')
    plt.vlines(onsets_selected_times, 0, 1, color='k', label='Onsets selected')
    plt.ylim(0, 3)
    plt.xlim(0, 10)
    plt.legend(frameon=True, framealpha=0.75, bbox_to_anchor=(1.15, 1));

def confusion(tn, fp, fn, tp):
    """Display a confusion matrix together with the accuracy, precision,
    recall, and F-measure.

    Arguments:
        tn (int): Number of true negatives.
        fp (int): Number of false positives.
        fn (int): Number of false negatives.
        tp (int): Number of true positives.
    """
    n = max(len(str(tn)), len(str(fp)), len(str(fn)), len(str(tp)))
    print(" " + (2 * n + 13) * "-")
    print(f'| tp: {tp:{n}} | fp: {fp:{n}} |')
    print(f'| fn: {fn:{n}} | tn: {tn:{n}} |')
    print(" " + (2 * n + 13) * "-")
    a, p, r, F = utils.measures(tn, fp, fn, tp)
    print(f' Accuracy: {a:.4f}')
    print(f'Precision: {p:.4f}')
    print(f'   Recall: {r:.4f}')
    print(f'F-measure: {F:.4f}')

def clicks(wav, beats, clicks_volume=8):
    r"""Returns a wav with clicks at the specified beats.

    Arguments:
        wav (1d numpy array): An audio wav at the sampling rate determined by
            `beatfinder.constants`.
        beats (1d numpy array): The beats in units of seconds.
        clicks_volume (float between 0 and 10): The volume of the clicks (it
            can also be set to 11).
    """

    clicks = librosa.clicks(times=beats,
                sr=constants.sr,
                length=len(wav))
    clicks *= clicks_volume / 10

    return wav + clicks
