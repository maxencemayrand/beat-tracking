import beatfinder
import numpy as np
from sys import argv

dataset_file = argv[1]

dataset = beatfinder.datasets.AudioBeatsDataset(file=dataset_file)
dataset.precompute()
print()
