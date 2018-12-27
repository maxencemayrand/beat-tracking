import matplotlib.pyplot as plt
import librosa
import numpy as np

from . import utils
from . import constants

def showdata(audiobeats, duration=None, offset=None):
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
    plt.figure(figsize=(16, 4))
    times = librosa.frames_to_time(np.arange(spec.shape[1]), constants.sr, constants.hl)
    freq = librosa.mel_frequencies(n_mels=constants.nb, fmin=constants.fm, htk=constants.htk)
    plt.pcolormesh(times, freq, spec)

def showprediction(ground_truth, predicted_beats, onsets_selected):
    onsets_selected_times = librosa.frames_to_time(onsets_selected, constants.sr, constants.hl)
    plt.figure(figsize=(16, 2))
    plt.vlines(ground_truth, 2, 3, color='g', label='Ground truth')
    plt.vlines(predicted_beats, 1, 2, color='b', label='Prediction')
    plt.vlines(onsets_selected_times, 0, 1, color='k', label='Onsets selected')
    plt.ylim(0, 3)
    plt.xlim(0, 10)
    plt.legend(frameon=True, framealpha=0.75, bbox_to_anchor=(1.15, 1));
    
def print_params(model):
    nb_params = 0
    print(" " + "-" * 58)
    for n, p in model.named_parameters():
        nb_p = p.shape.numel()
        nb_params += nb_p
        print(f'| {n:28} | {str(list(p.shape)):12} | {nb_p:10,} |')
    print(" " + "-" * 58)
    print(f"\nTotal number of parameters: {nb_params:,}")
    
def confusion(tn, fp, fn, tp):
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
    return wav + librosa.clicks(times=beats, sr=constants.sr, length=len(wav)) * clicks_volume / 10
    
    
    
    