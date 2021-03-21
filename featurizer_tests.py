import featurizer as ft
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import utils as ut
import librosa

# Plots to see if spectral centroid appears to be the weighted average frequency.
# To run, select and right-click and select "Run Selection/Line in Interactive Window"
data, fs = librosa.load('./train/train700.wav')
data = np.array(data)
data = ut.middle_n(data, 2000000)
sc = ft.calc_spectral_centroid(data, fs)
plt.plot(sc)
plt.show()