"""Default constants"""

sr = 22050    # sampling rate
hl = 256      # hop length
nb = 256      # number of frequency bins in the spectrograms
fm = 30.0     # lowest frequency in spectrograms
htk = False   # whether to use the HTK formula to compute spectrograms
dt = 0.05     # maximum distance between an onset and a beat

# Peak pick parameters (to be passed to librosa.peak_pick)
pre_max  = 3
post_max = 4
pre_avg  = 8
post_avg = 9
wait     = 2
delta    = 0.03
offset   = 1
