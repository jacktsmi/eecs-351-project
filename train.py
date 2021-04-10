import numpy as np
from scipy.io import wavfile
import sys
from scipy.fftpack import dct
import torch
import featurizer as ft
import utils as ut
from models import Net
from dataloader import MyDataset
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary

sc_net = Net()
sb_net = Net()
mfcc_net = Net()
chroma_net = Net()

criterion = nn.BCELoss()
sc_train = []
sb_train = []
mfcc_train = []
chroma_train = []
targets = [] # s

for ind in range(729):
    i = str(ind+1)
    sc = np.load("train_spectral_centroid/" + i + ".npy")
    sb = np.load("train_spectral_bandwidth/" + i + ".npy")
    mfcc = np.load("train_mfcc/" + i + ".npy")
    chroma = np.load("train_chroma/" + i + ".npy")
    sc_train.append(sc)
    sb_train.append(sb)
    mfcc_train.append(mfcc)
    chroma_train.append(chroma)
    i = int(i)
    if i < 221:
        label = [1, 0, 0, 0] # Happy
    elif i < 380:
        label = [0, 1, 0, 0] # Sad
    elif i < 549:
        label = [0, 0, 1, 0] # Calm
    else:
        label = [0, 0, 0, 1] # Hype
    targets.append(label)

sc_data = MyDataset(sc_train, targets)
sb_data = MyDataset(sb_train, targets)
mfcc_data = MyDataset(mfcc_train, targets)
chroma_data = MyDataset(chroma_train, targets)

batch_size = 10

num_epochs = 10

save_dir = "weights"

learning_rate = 0.0002

sc_loader = DataLoader(sc_data, batch_size=batch_size, shuffle=True)
sb_loader = DataLoader(sb_data, batch_size=batch_size, shuffle=True)
mfcc_loader = DataLoader(mfcc_data, batch_size=batch_size, shuffle=True)
chroma_loader = DataLoader(chroma_data, batch_size=batch_size, shuffle=True)

loss_traj = []

for epoch in range(num_epochs):
    print("Epoch #" + str(epoch+1) + "/" str(num_epochs))

    loss_epoch = 0

    # Another loop through batches