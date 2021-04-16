import numpy as np
from scipy.io import wavfile
import sys
from scipy.fftpack import dct
import torch
import featurizer as ft
import utils as ut
from models import MyNet, weights_init
from dataloader import MyDataset, ConcatDataset
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

# Test Neural Network First

print('Provide the full path to a song in WAV form:')
path = str(input())

fs, song = wavfile.read(path)

data = ut.middle_n(song, 1000000)

mfcc = ft.calc_mfcc(data, fs)
mfcc = mfcc.reshape(mfcc.shape[0]*mfcc.shape[1])

mfcc = torch.tensor(mfcc).double()

mfcc_net = MyNet(input_dim = mfcc.shape[0]).double()

mfcc_net.load_state_dict(torch.load('mfcc_net_model2.pt'))

out = mfcc_net(mfcc)

out_max, preds = torch.max(out, dim=0)
pred = preds.item()

if pred == 0:
    print('Neural network predicts the song is Happy')
if pred == 1:
    print('Neural network predicts the song is Sad')
if pred == 2:
    print('Neural network predicts the song is Calm')
if pred == 3:
    print('Neural network predicts the song is Hype')

# Next test KNN

knn_pred = ut.k_nn(path)

if knn_pred == 0:
    print('KNN predicts the song is Happy')
if knn_pred == 1:
    print('KNN predicts the song is Sad')
if knn_pred == 2:
    print('KNN predicts the song is Calm')
if knn_pred == 3:
    print('KNN predicts the song is Hype')

# Finally, SVM

svm_pred = ut.calc_svm(path)

if svm_pred == 0:
    print('SVM predicts the song is Happy')
if svm_pred == 1:
    print('SVM predicts the song is Sad')
if svm_pred == 2:
    print('SVM predicts the song is Calm')
if svm_pred == 3:
    print('SVM predicts the song is Hype')