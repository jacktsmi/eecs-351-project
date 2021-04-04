import numpy as np
from scipy.io import wavfile
import sys
from scipy.fftpack import dct

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

def calc_spectral_bandwidth(song,fs,sc,frame_size=1000):
    """
    Will calculate spectral bandwidth of a song. Should be called
    in featurize(). 
    Inputs:
        song : N, numpy array representing audio signal. Be sure to
        extract just one channel before passing into function. See
        featurizer_tests.py for example
        fs : sampling frequency of the song
        frame_size : width of the frame, in samples, of where we wish to
        calculate the spectral bandwidth.

        Outputs:
        spectral_bandwidth : N/frame_size length vector representing the spectral centroid
        value for each frame of the song.
    """
    spectral_bandwidth = np.zeros_like(sc)
    ind = 0

    for i in range(spectral_bandwidth.shape[0]):
        x = song[ind:ind+frame_size]
        magnitudes = np.abs(np.fft.rfft(x)) # magnitudes of positive frequencies
        length = len(x)
        freqs = np.abs(np.fft.fftfreq(length, 1.0/fs)[:length//2+1]) # positive frequencies
        spectral_bandwidth[i] = (np.sum(magnitudes*(freqs-sc[i])**2))**(1/2) # return weighted mean
        ind = ind+frame_size

    return spectral_bandwidth

def calc_mfcc(song, fs, frame_size=1000):
    """
    Will calculate MFCC's of a song. Should be called
    in featurize().
    """
    mfcc = np.zeros((int(song.shape[0] / frame_size)))
    pre_emphasis = 0.97
    emph_signal = np.append(song[0], song[1:] - pre_emphasis * song[:-1])
    frame_stride = 500 # Resulting in 500 sample overlap

    song_length = len(emph_signal)
    frame_length = int(round(frame_size))
    frame_step = int(round(frame_stride))
    num_frames = int(np.ceil(float(np.abs(song_length - frame_length)) / frame_step))

    pad_song_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_song_length - song_length))
    pad_song = np.append(emph_signal, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_song[indices.astype(np.int32, copy=False)]

    frames *= np.hamming(frame_length)

    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 * NFFT) * ((mag_frames) ** 2))

    nfilt = 40

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bin = np.floor((NFFT + 1) * hz_points / fs)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m-1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
    
    return mfcc
    

def calc_chroma(song):
    """
    Will calculate chroma of a song. Should be called
    in featurize().
    """

def featurize(song, fs, frame_size=1000):
    """
    Will featurize an audio signal into Spectral Centroid, Spectral Bandwidth,
    MFCC, and Chroma.

    Inputs:
        song : length N vector representing audio signal
    
    Outputs:
        feat_song : length 4 vector representing featurized audio
    """
    return song_feat