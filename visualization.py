import matplotlib.pyplot as plt
import librosa
import numpy as np
from constants import *

def showdata(spec, onsets, isbeat, beats, duration=5, offset=None):
    onsets_times = librosa.frames_to_time(onsets, sr, hl)

    onsets_selected = onsets[isbeat == 1]
    onsets_selected_times = librosa.frames_to_time(onsets_selected, sr, hl)

    onset_envelope = librosa.onset.onset_strength(S=spec)
    onset_envelope = onset_envelope / onset_envelope.max()

    times = librosa.frames_to_time(np.arange(len(onset_envelope)), sr, hl)

    total_duration = librosa.frames_to_time(spec.shape[1], sr, hl)
    if offset == None:
        offset = np.random.rand() * (total_duration - duration)

    plt.figure(figsize=(16, 8))

    plt.subplot(2, 1, 1)
    plt.plot(times, onset_envelope)
    plt.vlines(onsets_times, 0, 1, color='b', linestyles='--', alpha=0.75, label='Onsets')
    plt.vlines(onsets_selected_times, 0.7, 1, color='r', linestyles='-', label='Onsets selected')
    plt.vlines(beats, 1, 1.3, color='g', label='Beats recorded')
    plt.ylim(0, 1.3)
    plt.xlim(offset, offset + duration)
    plt.legend(frameon=True, framealpha=0.75, bbox_to_anchor=(1.15, 1));

    plt.subplot(2, 1, 2)
    freq = librosa.mel_frequencies(n_mels=nb, fmin=fm, htk=True)
    plt.pcolormesh(times, freq, spec)
    plt.xlim(offset, offset + duration);

def showspec(spec):
    plt.figure(figsize=(16, 4))
    times = librosa.frames_to_time(np.arange(spec.shape[1]), sr, hl)
    freq = librosa.mel_frequencies(n_mels=nb, fmin=fm, htk=htk)
    plt.pcolormesh(times, freq, spec)
    
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
    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
    accuracy  = (tp + tn) / (tp + tn + fp + fn)
    print(f'Precision: {precision:.4f}')
    print(f'   Recall: {recall:.4f}')
    print(f' Accuracy: {accuracy:.4f}')
    
    
    
    
    
    