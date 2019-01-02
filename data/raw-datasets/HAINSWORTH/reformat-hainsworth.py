import os
import numpy as np
import librosa

beats_path = './beats/'

if not os.path.exists(beats_path):
    os.makedirs(beats_path)

f = open('hainsworth/data.txt')
g = open('audio_files.txt', 'w')

for i, line in enumerate(f.readlines()):
    if i >= 13:
        data = [e.strip('\t\n ') for e in line.split('<sep>')]
        wav_file   = data[0]
        artist     = data[1]
        title      = data[2]
        bpm1       = data[3]
        style      = data[4]
        bigstyle   = data[5]
        tempocond  = data[6]
        difficulty = data[7]
        subdiv     = data[8]
        bpm2       = data[9]
        beats      = data[10]
        downbeats  = data[11]

        name         = wav_file.split('.')[0]
        beats_file   = os.path.join(beats_path, f'{name}.beats')
        beats_frames = np.array([float(n) for n in beats.split(',')])
        beats_times  = librosa.samples_to_time(beats_frames, sr=44100)
        np.savetxt(beats_file, beats_times, fmt='%.5f')
        g.write(f'./hainsworth/wavs/{name}.wav\n')
        print(name)
f.close()
g.close()
