import featurizer as ft
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import utils as ut

# Plots to see if spectral centroid appears to be the weighted average frequency.
# To run, select and right-click and select "Run Selection/Line in Interactive Window"
fs, data = wavfile.read('./train/train1.wav')
data = np.array(data)
data = ut.middle_n(data[:, 0], 2000000)
sc = ft.calc_spectral_centroid(data, fs)
plt.plot(sc)
plt.show()