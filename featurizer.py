import numpy as np
from scipy.io import wavfile
import sys

def calc_spectral_centroid(song, fs, frame_size=1000):
    """
    Will calculate spectral centroid value of a song. Should be called
    in featurize().
    Inputs:
        song : N, numpy array representing audio signal. Be sure to
        extract just one channel before passing into function. See
        featurizer_tests.py for example.
        fs : sampling frequency of the song
        frame_size : width of the frame, in samples, of where we wish to
        calculate the spectral centroid.
    
    Outputs:
        spectral_centroid : N/frame_size length vector representing the spectral centroid
        value for each frame of the song.
    """
    spectral_centroid = np.zeros((int(song.shape[0] / frame_size)))
    ind = 0
    for i in range(spectral_centroid.shape[0]):
        x = song[ind:ind+frame_size]
        magnitudes = np.abs(np.fft.rfft(x)) # magnitudes of positive frequencies
        length = len(x)
        freqs = np.abs(np.fft.fftfreq(length, 1.0/fs)[:length//2+1]) # positive frequencies
        spectral_centroid[i] = np.sum(magnitudes*freqs) / np.sum(magnitudes) # return weighted mean
        ind = ind+frame_size

    return spectral_centroid

def calc_spectral_bandwidth(song):
    """
    Will calculate spectral bandwidth of a song. Should be called
    in featurize().
    """

def calc_mfcc(song):
    """
    Will calculate MFCC's of a song. Should be called
    in featurize().
    """

def calc_chroma(song):
    """
    Will calculate chroma of a song. Should be called
    in featurize().
    """

def featurize(song):
    """
    Will featurize an audio signal into Spectral Centroid, Spectral Bandwidth,
    MFCC, and Chroma.

    Inputs:
        song : length N vector representing audio signal
    
    Outputs:
        feat_song : length 4 vector representing featurized audio
    """