import featurizer as ft
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import utils as ut
import librosa

# Plots to see if spectral centroid appears to be the weighted average frequency.
# To run, select and right-click and select "Run Selection/Line in Interactive Window"
fs1, data1 = wavfile.read('./train/train500.wav') # Calm Song
data1 = data1[:, 0] # Take only one channel
data1 = np.array(data1)
data1 = ut.middle_n(data1, 2000000)
sc1 = ft.calc_spectral_centroid(data1, fs1)
fs2, data2 = wavfile.read('./train/train700.wav') # Hype Song
data2 = data2[:, 0] # Take only one channel
data2 = np.array(data2)
data2 = ut.middle_n(data2, 2000000)
sc2 = ft.calc_spectral_centroid(data2, fs2)
#plt.plot(sc1, 'b')
#plt.plot(sc2, 'r')
#plt.show()

# MFCC
mfcc = ft.calc_mfcc(data1, fs1)
mfcc = np.array(mfcc)
plt.pcolormesh(mfcc.T)
plt.show()
mfcc2 = ft.calc_mfcc(data2, fs2)
mfcc2 = np.array(mfcc2)
plt.pcolormesh(mfcc2.T)
plt.show()