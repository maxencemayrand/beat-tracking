import os
import numpy as np

beats_path = './beats/'
if not os.path.exists(beats_path):
    os.makedirs(beats_path)

def load_beats(file):
    lines = open(file).readlines()
    beats = np.zeros(len(lines))
    for k in range(len(lines)):
        beats[k] = float(lines[k].split()[0])
    return beats

names = []
with open('BallroomData/allBallroomFiles') as f:
    for line in f.readlines():
        file = line.strip('\n')[2:]
        names.append(os.path.splitext(os.path.basename(file))[0])

for i, n in enumerate(names):
    annotation_path = f'BallroomAnnotations/{n}.beats'
    beats = load_beats(annotation_path)
    path = f'{beats_path}{n}.beats'
    np.savetxt(path, beats, fmt='%.5f')
    print(f'{i+1}/{len(names)} | {path}')
