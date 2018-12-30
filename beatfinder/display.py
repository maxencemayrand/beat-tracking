import matplotlib.pyplot as plt
import librosa
import numpy as np

from . import utils
from . import constants

def showdata(audiobeats, duration=None, offset=None):
    r"""Displays the data pointed by an AudioBeats object. It shows the onset envelope, the onsets, the onsets selected as beats, the ground truth beats, and the spectrogram.

    Arguments:
        audiobeats (AudioBeats): The `AudioBeats` object to display.
        duration (float): To see a smaller portion of the data, this can be set
            to a lower value than the duration of `audiobeats`.
        offset (float): Display only the data starting from `offset` (combine)
            with `duration` to see a smaller portion of the data).
    """

    spec, onsets, isbeat, beats = audiobeats.get_data()
    if duration == None:
        duration = audiobeats.duration
    if offset == None:
        offset = 0

    onsets_times = librosa.frames_to_time(onsets, constants.sr, constants.hl)

    onsets_selected = onsets[isbeat == 1]
    onsets_selected_times = librosa.frames_to_time(onsets_selected, constants.sr, constants.hl)

    onset_envelope = utils.onset_strength(spec=spec)

    times = librosa.frames_to_time(np.arange(len(onset_envelope)), constants.sr, constants.hl)

    total_duration = librosa.frames_to_time(spec.shape[1], constants.sr, constants.hl)

    plt.figure(figsize=(16, 8))

    plt.subplot(2, 1, 1)
    plt.plot(times, onset_envelope)
    plt.vlines(onsets_times, 0, 1, color='b', linestyles='--', alpha=0.75, label='Onsets')
    plt.vlines(onsets_selected_times, 0.7, 1, color='r', linestyles='-', label='Onsets selected')
    plt.vlines(beats, 1, 1.3, color='g', label='Ground truth')
    plt.ylim(0, 1.3)
    plt.xlim(offset, offset + duration)
    plt.legend(frameon=True, framealpha=0.75, bbox_to_anchor=(1.15, 1));

    plt.subplot(2, 1, 2)
    freq = librosa.mel_frequencies(n_mels=constants.nb, fmin=constants.fm, htk=constants.htk)
    plt.pcolormesh(times, freq, spec)
    plt.xlim(offset, offset + duration);


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
