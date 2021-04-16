import pathlib
import os
import math
import numpy as np
from scipy.io import wavfile
import featurizer as ft
from scipy import stats
from sklearn import svm

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

def save_train(out_path, train_path, frame_size=1000):
    """
    Will featurize every training file and save the numpy arrays in a text
    file.

    Inputs:
        out_path : string of path where files should be saved.

        train_path : string of path name where training files are

        frame_size : size of frame to be used in feature calculation
    
    Outputs:
        None
    """
    for path in pathlib.Path(train_path).iterdir():
        fs, song = wavfile.read(train_path + '/' + path)
        song = song[:, 0]
        song = middle_n(song, 2000000)
        song_feat = ft.featurize(song, fs, frame_size)
        np.save(out_path + path, song_feat)

def load_train(load_path):
    """
    Loads in training data as saved by the save_train function and stores
    the data in a 2d numpy array.

    Inputs:
        load_path : string of path where we load in the data
    
    Outputs:
        train_data : num_train_samps x input_dim numpy array containing featurized
        training data
    """
    num_samps = 729
    input_dim = 8000
    train_data = np.zeros((num_samps, input_dim))
    index = 0
    for path in pathlib.Path(load_path).iterdir():
        train_data[index, :] = np.load(path)
    return train_data

def k_nn(path):
    fs, data1 = wavfile.read(path)
    if data1.ndim > 1:
        data1 = data1[:, 0]
    data = middle_n(data1, 1000000)
    sc = ft.calc_spectral_centroid(data, fs)
    sb = ft.calc_spectral_bandwidth(data, fs, sc)
    mfcc = ft.calc_mfcc(data, fs)
    data2 = middle_n(data1, 25000)
    chroma = ft.calc_chroma(data2, fs)

    train_sc = np.zeros((250, 160*4))
    train_sb = np.zeros((250, 160*4))
    train_mfcc = np.zeros((5976, 160*4))
    train_chroma = np.zeros((25*12, 160*4))

    train_targets = []

    accuracies = []

    train_i = 0

    for ind in range(1, 730):
        i = str(ind)
        sc = np.load("train_spectral_centroid/" + i + ".npy")
        sb = np.load("train_spectral_bandwidth/" + i + ".npy")
        mfcc = np.load("train_mfcc/" + i + ".npy")
        chroma = np.load("train_chroma/" + i + ".npy")

        if 0 <= ind <= 12+147:
            train_sc[:,train_i] = sc
            train_sb[:,train_i] = sb
            train_mfcc[:,train_i] = mfcc
            train_chroma[:,train_i] = chroma
            train_i += 1
            label = 0 # Happy
            train_targets.append(label)
        elif 220 <= ind <= 232+147:
            train_sc[:,train_i] = sc
            train_sb[:,train_i] = sb
            train_mfcc[:,train_i] = mfcc
            train_chroma[:,train_i] = chroma
            train_i += 1
            label = 1 # Sad
            train_targets.append(label)
        elif 380 <= ind <= 392+147:
            train_sc[:,train_i] = sc
            train_sb[:,train_i] = sb
            train_mfcc[:,train_i] = mfcc
            train_chroma[:,train_i] = chroma
            train_i += 1
            label = 2 # Calm
            train_targets.append(label)
        elif 549 <= ind <= 561+147:
            train_sc[:,train_i] = sc
            train_sb[:,train_i] = sb
            train_mfcc[:,train_i] = mfcc
            train_chroma[:,train_i] = chroma
            train_i += 1
            label = 3 # Hype
            train_targets.append(label)

    k = 40

    sc_diff = train_sc - np.tile(sc, (160*4,1)).T
    sb_diff = train_sb - np.tile(sb, (160*4,1)).T
    mfcc_diff = train_mfcc - np.tile(mfcc, (160*4,1)).T
    chroma_diff = train_chroma - np.tile(chroma, (160*4,1)).T

    sc_norm = np.linalg.norm(sc_diff,axis = 0)
    sb_norm = np.linalg.norm(sb_diff,axis = 0)
    mfcc_norm = np.linalg.norm(mfcc_diff,axis = 0)
    chroma_norm = np.linalg.norm(chroma_diff,axis = 0)

    sc_ind = np.argsort(sc_norm)[:k]
    sb_ind = np.argsort(sb_norm)[:k]
    mfcc_ind = np.argsort(mfcc_norm)[:k]
    chroma_ind = np.argsort(chroma_norm)[:k]

    sc_labels = np.array([train_targets[i] for i in sc_ind])
    sb_labels = np.array([train_targets[i] for i in sb_ind])
    mfcc_labels = np.array([train_targets[i] for i in mfcc_ind])
    chroma_labels = np.array([train_targets[i] for i in chroma_ind])

    out = stats.mode(np.concatenate((sc_labels, sb_labels, mfcc_labels, chroma_labels)))[0][0]

    return out

def calc_svm(path):
    train_sc = np.zeros((250, 639))
    train_sb = np.zeros((250, 639))
    train_mfcc = np.zeros((5976, 639))
    train_chroma = np.zeros((25*12, 639))

    train_targets = []

    train_i = 0

    for ind in range(1, 730):
        i = str(ind)
        sc = np.load("train_spectral_centroid/" + i + ".npy")
        sb = np.load("train_spectral_bandwidth/" + i + ".npy")
        mfcc = np.load("train_mfcc/" + i + ".npy")
        chroma = np.load("train_chroma/" + i + ".npy")

        if 0 <= ind <= 12+147:
            train_sc[:,train_i] = sc
            train_sb[:,train_i] = sb
            train_mfcc[:,train_i] = mfcc
            train_chroma[:,train_i] = chroma
            train_i += 1
            label = 0 # Happy
            train_targets.append(label)
        elif 220 <= ind <= 232+147:
            train_sc[:,train_i] = sc
            train_sb[:,train_i] = sb
            train_mfcc[:,train_i] = mfcc
            train_chroma[:,train_i] = chroma
            train_i += 1
            label = 1 # Sad
            train_targets.append(label)
        elif 380 <= ind <= 392+147:
            train_sc[:,train_i] = sc
            train_sb[:,train_i] = sb
            train_mfcc[:,train_i] = mfcc
            train_chroma[:,train_i] = chroma
            train_i += 1
            label = 2 # Calm
            train_targets.append(label)
        elif 549 <= ind <= 561+147:
            train_sc[:,train_i] = sc
            train_sb[:,train_i] = sb
            train_mfcc[:,train_i] = mfcc
            train_chroma[:,train_i] = chroma
            train_i += 1
            label = 3 # Hype
            train_targets.append(label)

    clf_sc = svm.SVC(class_weight='balanced')
    clf_sb = svm.SVC(class_weight='balanced')
    clf_mfcc = svm.SVC(class_weight='balanced')
    clf_chroma = svm.SVC(class_weight='balanced')

    clf_sc.fit(train_sc.T, train_targets)
    clf_sb.fit(train_sb.T, train_targets)
    clf_mfcc.fit(train_mfcc.T, train_targets)
    clf_chroma.fit(train_chroma.T, train_targets)

    sc_preds = clf_sc.predict(sc.reshape(1, -1))
    sb_preds = clf_sb.predict(sb.reshape(1, -1))
    mfcc_preds = clf_mfcc.predict(mfcc.reshape(1, -1))
    chroma_preds = clf_chroma.predict(chroma.reshape(1, -1))

    return stats.mode(np.concatenate((sc_preds, sb_preds, mfcc_preds, chroma_preds)))[0][0]