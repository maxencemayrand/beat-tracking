import os
import numpy as np
import pandas as pd
import jams

beats_path = './beats/'
if not os.path.exists(beats_path):
    os.makedirs(beats_path)

def load_beats(file):
    df = jams.load(file)['annotations'][0].to_dataframe()
    beats = df['time'].values
    return beats

styles = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

names = []
files = []
f = open('audio_files.txt', 'w')
for s in styles:
    for n in range(100):
        name = f'{s}.{n:05}'
        names.append(name)
        file = f'./genres/{s}/{name}.au'
        files.append(file)
        f.write(file + '\n')
f.close()

for i, n in enumerate(names):
    annotation_path = f'rhythm/jams/{n}.wav.jams'
    beats = load_beats(annotation_path)
    path = f'{beats_path}{n}.beats'
    np.savetxt(path, beats, fmt='%.5f')
    print(f'{i+1}/{len(names)} | {path}')
