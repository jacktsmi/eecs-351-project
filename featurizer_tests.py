import featurizer as ft
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import utils as ut
import librosa
import librosa.display

# Plots to see if spectral centroid appears to be the weighted average frequency.
# To run, select and right-click and select "Run Selection/Line in Interactive Window"
fs1, data1 = wavfile.read('./train/train500.wav') # Calm Song
data1 = data1[:, 0] # Take only one channel
data1 = np.array(data1)
data1 = ut.middle_n(data1, 20000)
"""
sc1 = ft.calc_spectral_centroid(data1, fs1)
fs2, data2 = wavfile.read('./train/train700.wav') # Hype Song
data2 = data2[:, 0] # Take only one channel
data2 = np.array(data2)
data2 = ut.middle_n(data2, 2000000)
sc2 = ft.calc_spectral_centroid(data2, fs2)
calm_sc = plt.plot(sc1, 'b', label='Calm Song')
hype_sc = plt.plot(sc2, 'r', label='Hype Song')
plt.title('Spectral Centroid')
plt.legend()
plt.show()

"""
# Checking chroma features
coeff, coeff_freq = ft.calc_chroma(data1, fs1)
spec = plt.figure(figsize=(10,4))
plt.imshow(10*np.log10(coeff))
plt.clim([-30, 30])
plt.xlim([1, 20])
plt.ylim([0, coeff_freq[19][999]])
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
cbar = plt.colorbar()
cbar.set_label('Magnitude (dB)')

# Plot Spectrogram
"""spec = plt.figure(figsize=(10, 4))
plt.imshow(10 * np.log10(magn), origin='lower', aspect='auto', cmap='gray_r',
           extent=[T_coef[0], T_coef[-1], F_coef[0], F_coef[-1]])
plt.clim([-30, 30])
plt.ylim([0, freqs[999]])
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
cbar = plt.colorbar()
cbar.set_label('Magnitude (dB)')

y, sr = librosa.load('./train/train500.wav')

S = np.abs(librosa.stft(y))
chroma = librosa.feature.chroma_stft(S = S,sr = sr)

plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram')
plt.tight_layout()

"""
# Plotting Spectral Bandwidth
"""
sb1 = ft.calc_spectral_bandwidth(data1, fs1, sc1)
sb2 = ft.calc_spectral_bandwidth(data2, fs2, sc2)

calm = plt.plot(sb1, 'b', label='Calm Song')
hype = plt.plot(sb2, 'r', label='Hype Song')
plt.title('Spectral Bandwidth')
plt.legend()
plt.show()

# MFCC
mfcc = ft.calc_mfcc(data1, fs1)
mfcc = np.array(mfcc)
plt.pcolormesh(mfcc.T)
plt.title('Calm Song MFCC')
plt.show()
mfcc2 = ft.calc_mfcc(data2, fs2)
mfcc2 = np.array(mfcc2)
plt.pcolormesh(mfcc2.T)
plt.title('Hype Song MFCC')
plt.show()
"""