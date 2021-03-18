import numpy as np

def featurize(song):
    """
    Will featurize an audio signal into Spectral Centroid, Spectral Bandwidth,
    MFCC, and Chroma.

    Inputs:
        song : length N vector representing audio signal
    
    Outputs:
        feat_song : length 4 vector representing featurized audio
    """