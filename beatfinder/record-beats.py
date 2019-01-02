import time
from sys import argv
import os

dir = argv[1]
start = time.time()

beats = []
while True:
    i = input()
    beats.append(time.time() - start)
    if i == 'q':
        break
    print(f'{beats[-1]:.4f}', end=' ')

f = open(os.path.join(dir, 'beats.txt'), 'w')
for b in beats[:-1]:
    f.write(f'{b:.4f}\n')
f.close()
