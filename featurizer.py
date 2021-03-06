import numpy as np
import scipy
from scipy.io import wavfile
import sys
import math
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
    # Weighted Average of Frequencies by their Magnitudes for each frame (center of the signal)
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

    # Standard Deviation of the Signal (Power of the TF around the center frequency)
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

    Inputs:
        song : N, numpy array representing audio signal. Be sure to
        extract just one channel before passing into function. See
        featurizer_tests.py for example

        fs : sampling frequency as output by wavfile.read

        frame_size : size of window for mfcc calculation
    
    Outputs:
        mfcc : (N/frame_size*2 - 2, 12) numpy array of mfcc values
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
    

def calc_chroma(song, fs, frame_size=1000):
    """
    Will calculate chroma of a song. Should be called
    in featurize().

    Inputs:
        song: length N vector representing audio signal
        fs: sampling frequency
        frame_size = 1000 (by default)

    Outputs:
        mag_chroma: Output 2d vector that computes chromagram

    Procedure:
    1. Compute the linear spectrogram of the audio signal
        a. 1000 frames (window on each frame) -> perform FFT on each one = STFT
        b. Resulting coefficient from FFT gives us the magnitude
        c. x-axis is in time (one magnitude per (audio sample length)/1000 s)
        d. y-axis has one magnitude associated with each frequency (length of FFT) for each frame
    2. Change axis scale to log frequency (Hz to p-values)
    3. Add up corresponding chroma labels {C, C#, ..., B} within the frequency range to compute chromagram
    """
    """
    magnitudes = np.zeros((int(song.shape[0] / frame_size)))
    ind = 0
    for i in range(magnitudes.shape[0]):
        x = song[ind:ind + frame_size]
        magnitudes[i] = np.abs(np.fft.rfft(x))  # magnitudes of positive frequencies
        length = len(x)
        freqs = np.abs(np.fft.fftfreq(length, 1.0 / fs)[:length // 2 + 1])  # positive frequencies
        ind = ind + frame_size
    Y = np.abs(magnitudes)**2
    """
    # w = scipy.hamming(frame) - Ignore hamming window for now

    # C[0] returns an array of magnitudes whose indices are frequencies at frame 0
    # C[i][j] - i represents frame number, j represents frequency, value is magnitude

    songLen = song.shape[0] # 2000000 samples
    num_frames = songLen//frame_size # x-axis on Chromagram
    ind = 0 # Essentially hopping parameter
    C = np.zeros((num_frames, frame_size)) # Magnitudes of Fourier coefficients (see above)
    C_freq = np.zeros((num_frames, frame_size)) # Magnitudes of each frequency at each time step
    # Loops through all the frames to compute Fourier coefficients and frequencies
    for i in range(0, num_frames):
        curr_mag = np.fft.rfft(song[ind: ind + frame_size])
        curr_mag_freqs = np.fft.rfftfreq(frame_size, d=1/fs)
        for j in range(0, curr_mag.shape[0]):
            C[i][j] = np.abs(curr_mag[j])**2
            C_freq[i][j] = curr_mag_freqs[j]
        ind += frame_size

    # Lowest is C1, highest is G9 all with respect to A4 = 440 Hz (p = 69)

    lower = np.zeros(128)
    upper = np.zeros(128)
    # Finds upper and lower bounds for each MIDI pitch
    for p in range(24, 127):
        lower[p] = 2 ** ((p - 0.5 - 69) / 12) * 440
        upper[p] = 2 ** ((p + 0.5 - 69) / 12) * 440

    # Loops through all the frames, time steps, fits magnitude under the pitch within
    # which it fits
    mag_pitch = np.zeros((num_frames, 128))
    for i in range(0, num_frames):
        for j in range(0, frame_size):
            for p in range(24, 127):
                if lower[p] <= C_freq[i][j] < upper[p]:
                    mag_pitch[i][p] += C[i][j]
                    break

    # Matches the chroma labels to the MIDI pitches
    mag_chroma = np.zeros((num_frames, 12))
    for i in range(0, num_frames):
        for p in range(24, 127):
            mag_chroma[i][(p - 60) % 12] += mag_pitch[i][p]
        for p in range(0, 12):
            mag_chroma[i][p] = 10*np.log10(mag_chroma[i][p])

    # First index is time-step, second-index is chroma label starting at C and ending at B
    # Value of array is magnitude of that point on chromagram
    return mag_chroma

    # chroma = np.fmod(np.round(np.log2(X / 440) * 12), 12)

    # f, t, STFT = scipy.signal.stft(song, fs)
    
    # f = Array of sample frequencies
    # t = Array of segment times.
    # STFT = STFT of song (complex, plot magnitude)


def featurize(song, fs, frame_size=1000):
    """
    Will featurize an audio signal into Spectral Centroid, Spectral Bandwidth,
    MFCC, and Chroma.

    Inputs:
        song : length N vector representing audio signal
    
    Outputs:
        feat_song : length 4 vector representing featurized audio
    """

    sc = calc_spectral_centroid(song, fs)
    sb = calc_spectral_bandwidth(song, fs, sc)
    mfcc = calc_mfcc(song, fs)
    mfcc.reshape(mfcc.shape[0]*mfcc.shape[1])
    

    features = [sc,sb,mfcc]

    return features