sr = 22050    # sampling rate
hl = 256      # hop length
nb = 256      # number of frequency bins in the spectrograms
fm = 30.0     # lowest frequency in spectrograms
htk = False   # 
dt = 0.05     # maximum distance between an onset and a beat

# Peak pick parameters
pre_max = 3
post_max = 4
delta = 0.03

# length of one sample in each dataset.
#from librosa import time_to_frames
#default_sample_length = 10    # in seconds
#default_nb_frames_per_sample = time_to_frames(sample_length, sr, hl)
