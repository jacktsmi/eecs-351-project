import featurizer as ft
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import utils as ut
import librosa
import librosa.display

# Plots to see if spectral centroid appears to be the weighted average frequency.
# To run, select and right-click and select "Run Selection/Line in Interactive Window"
fs1, data1 = wavfile.read('./train/train100.wav') # Happy Song
data1 = data1[:, 0] # Take only one channel
data1 = np.array(data1)
data1 = ut.middle_n(data1, 250000)

fs2, data2 = wavfile.read('./train/train300.wav') # Sad Song
data2 = data2[:, 0] # Take only one channel
data2 = np.array(data2)
data2 = ut.middle_n(data2, 250000)

fs3, data3 = wavfile.read('./train/train498.wav') # Calm Song
data3 = data3[:, 0] # Take only one channel
data3 = np.array(data3)
data3 = ut.middle_n(data3, 250000)

fs4, data4 = wavfile.read('./train/train700.wav') # Hype Song
data4 = data4[:, 0] # Take only one channel
data4 = np.array(data4)
data4 = ut.middle_n(data4, 250000)

sc1 = ft.calc_spectral_centroid(data1, fs1)
sc2 = ft.calc_spectral_centroid(data2, fs2)
sc3 = ft.calc_spectral_centroid(data3, fs3)
sc4 = ft.calc_spectral_centroid(data4, fs4)

fig1, ax1 = plt.subplots(figsize=((14, 7)))
happy_sc = ax1.plot(sc1, 'r', label='Happy Song')
sad_sc = ax1.plot(sc2, 'b', label='Sad Song')
calm_sc = ax1.plot(sc3, 'g', label='Calm Song')
hype_sc = ax1.plot(sc4, 'c', label='Hype Song')
plt.title('Spectral Centroid', fontsize=20)
plt.xlabel('Frame Number (~0.02 s/frame)', fontsize=15)
plt.ylabel('Frequency (Hz)', fontsize=15)
plt.legend(loc='upper right')
plt.savefig('Spectral_Centroid_Moods')
plt.show()

sb1 = ft.calc_spectral_bandwidth(data1, fs1, sc1)
sb2 = ft.calc_spectral_bandwidth(data2, fs2, sc2)
sb3 = ft.calc_spectral_bandwidth(data3, fs3, sc3)
sb4 = ft.calc_spectral_bandwidth(data4, fs4, sc4)

fig2, ax2 = plt.subplots(figsize=((14, 7)))
happy_sb = ax2.plot(sb1, 'r', label='Happy Song')
sad_sb = ax2.plot(sb2, 'b', label='Sad Song')
calm_sb = ax2.plot(sb3, 'g', label='Calm Song')
hype_sb = ax2.plot(sb4, 'c', label='Hype Song')
plt.title('Spectral Bandwidth', fontsize=20)
plt.xlabel('Frame Number (~0.02 s/frame)', fontsize=15)
plt.ylabel('Frequency (Hz)', fontsize=15)
plt.legend(loc='upper right')
plt.savefig('Spectral_Bandwidth_Moods')

mfcc1 = ft.calc_mfcc(data1, fs1)
mfcc2 = ft.calc_mfcc(data2, fs2)
mfcc3 = ft.calc_mfcc(data3, fs3)
mfcc4 = ft.calc_mfcc(data4, fs4)

fig3, ax3 = plt.subplots(2,2,figsize=(20, 15))
ax3[0][0].pcolormesh(mfcc1.T)
ax3[0][0].set_title('Happy Song', fontsize=20)
ax3[0][0].set_xlabel('Frame Number', fontsize=15)
ax3[0][0].set_ylabel('MFCC Index', fontsize=15)
ax3[0][0].set_yticks(np.arange(0.5, 12.5, 1))
ax3[0][0].set_yticklabels('1 2 3 4 5 6 7 8 9 10 11 12'.split())

ax3[0][1].pcolormesh(mfcc2.T)
ax3[0][1].set_title('Sad Song', fontsize=20)
ax3[0][1].set_xlabel('Frame Number', fontsize=15)
ax3[0][1].set_ylabel('MFCC Index', fontsize=15)
ax3[0][1].set_yticks(np.arange(0.5, 12.5, 1))
ax3[0][1].set_yticklabels('1 2 3 4 5 6 7 8 9 10 11 12'.split())

ax3[1][0].pcolormesh(mfcc3.T)
ax3[1][0].set_title('Calm Song', fontsize=20)
ax3[1][0].set_xlabel('Frame Number', fontsize=15)
ax3[1][0].set_ylabel('MFCC Index', fontsize=15)
ax3[1][0].set_yticks(np.arange(0.5, 12.5, 1))
ax3[1][0].set_yticklabels('1 2 3 4 5 6 7 8 9 10 11 12'.split())

im_mfcc = ax3[1][1].pcolormesh(mfcc4.T)
ax3[1][1].set_title('Hype Song', fontsize=20)
ax3[1][1].set_xlabel('Frame Number', fontsize=15)
ax3[1][1].set_ylabel('MFCC Index', fontsize=15)
ax3[1][1].set_yticks(np.arange(0.5, 12.5, 1))
ax3[1][1].set_yticklabels('1 2 3 4 5 6 7 8 9 10 11 12'.split())

cbar_mfcc = fig3.colorbar(im_mfcc, ax=ax3)
fig3.suptitle('MFCC', fontsize=32)
cbar_mfcc.set_label('Magnitude (dB)', fontsize=20)
plt.savefig('MFCC_Moods')


chroma1 = ft.calc_chroma(data1, fs1)
chroma2 = ft.calc_chroma(data2, fs2)
chroma3 = ft.calc_chroma(data3, fs3)
chroma4 = ft.calc_chroma(data4, fs4)

fig4, ax4 = plt.subplots(2,2,figsize=(20, 15))

ax4[0][0].pcolormesh(chroma1.T)
ax4[0][0].set_title('Happy Song', fontsize=20)
ax4[0][0].set_xlabel('Frame Number', fontsize=15)
ax4[0][0].set_ylabel('Chroma', fontsize=15)
ax4[0][0].set_yticks(np.arange(0.5, 12.5, 1))
ax4[0][0].set_yticklabels('C C# D D# E F F# G G# A A# B'.split())

ax4[0][1].pcolormesh(chroma2.T)
ax4[0][1].set_title('Sad Song', fontsize=20)
ax4[0][1].set_xlabel('Frame Number', fontsize=15)
ax4[0][1].set_ylabel('Chroma', fontsize=15)
ax4[0][1].set_yticks(np.arange(0.5, 12.5, 1))
ax4[0][1].set_yticklabels('C C# D D# E F F# G G# A A# B'.split())

ax4[1][0].pcolormesh(chroma3.T)
ax4[1][0].set_title('Calm Song', fontsize=20)
ax4[1][0].set_xlabel('Frame Number', fontsize=15)
ax4[1][0].set_ylabel('Chroma Label', fontsize=15)
ax4[1][0].set_yticks(np.arange(0.5, 12.5, 1))
ax4[1][0].set_yticklabels('C C# D D# E F F# G G# A A# B'.split())

im_chroma = ax4[1][1].pcolormesh(chroma4.T)
ax4[1][1].set_title('Hype Song', fontsize=20)
ax4[1][1].set_xlabel('Frame Number', fontsize=15)
ax4[1][1].set_ylabel('Chroma Label', fontsize=15)
ax4[1][1].set_yticks(np.arange(0.5, 12.5, 1))
ax4[1][1].set_yticklabels('C C# D D# E F F# G G# A A# B'.split())

fig4.suptitle('Chromagram', fontsize=32)
cbar = fig4.colorbar(im_chroma, ax=ax4)
cbar.set_label('Magnitude (dB)', fontsize=20)
plt.savefig('Chromagram_Moods')