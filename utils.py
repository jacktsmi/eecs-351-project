import pathlib
import os
import math
import numpy as np
from scipy.io import wavfile
import featurizer as ft

def rename_data(dir_path, start_ind=1):
    """
    Renames all files in dir_path to "trainX" where X is a number.

    Inputs:
        dir_path : Path to the directory which has the files we wish to rename
        (For our purposes this will likely be the folder with either the train
        data or test data)

        start_ind : index to start the naming with before iterating
    """
    index = start_ind
    for path in pathlib.Path(dir_path).iterdir():
        if path.is_file():
            directory = path.parent
            old_extension = path.suffix
            new_name = "train" + str(index) + old_extension
        path.rename(pathlib.Path(directory, new_name))
        index = index + 1

def middle_n(vector, n):
    """
    Returns the middle n (n even) elements of a vector of length >=n.

    Inputs:
        vector : length >=n numpy array
    """
    mid_index = math.floor(vector.shape[0] / 2)
    return vector[mid_index - int(n / 2):mid_index + int(n / 2)]

def save_train(out_path, train_path, fs, frame_size=1000):
    """
    Will featurize every training file and save the numpy arrays in a text
    file.

    Inputs:
        out_path : string of path where files should be saved.
    """
    for path in pathlib.Path(train_path).iterdir():
        fs, song = wavfile.read(train_path + '/' + path)
        song = song[:, 0]
        song = middle_n(song, 2000000)
        song_feat = ft.featurize(song, fs, frame_size)
        np.save(out_path + path, song_feat)

def load_train(load_path):
    num_samps = 729
    input_dim = 8000
    train_data = np.zeros((num_samps, input_dim))
    index = 0
    for path in pathlib.Path(load_path).iterdir():
        train_data[index, :] = np.load(path)
    return train_data