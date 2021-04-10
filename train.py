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
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary

sc_train = []
sb_train = []
mfcc_train = []
chroma_train = []
sc_test = []
sb_test = []
mfcc_test = []
chroma_test = []
train_targets = []
test_targets = []

for ind in range(1, 730):
    i = str(ind)
    sc = np.load("train_spectral_centroid/" + i + ".npy")
    sb = np.load("train_spectral_bandwidth/" + i + ".npy")
    mfcc = np.load("train_mfcc/" + i + ".npy")
    chroma = np.load("train_chroma/" + i + ".npy")
    if 209 < ind < 221: # or 368 < ind < 380 or 537 < ind < 549 or 718 < ind < 730:
        sc_test.append(sc)
        sb_test.append(sb)
        mfcc_test.append(mfcc)
        chroma_test.append(chroma)
        label = 0 # Happy
        test_targets.append(label)
    elif 368 < ind < 380:
        sc_test.append(sc)
        sb_test.append(sb)
        mfcc_test.append(mfcc)
        chroma_test.append(chroma)
        label = 1 # Sad
        test_targets.append(label)
    elif 537 < ind < 549:
        sc_test.append(sc)
        sb_test.append(sb)
        mfcc_test.append(mfcc)
        chroma_test.append(chroma)
        label = 2 # Calm
        test_targets.append(label)
    elif 718 < ind < 730:
        sc_test.append(sc)
        sb_test.append(sb)
        mfcc_test.append(mfcc)
        chroma_test.append(chroma)
        label = 3 # Hype
        test_targets.append(label)
    elif ind <= 209:
        sc_train.append(sc)
        sb_train.append(sb)
        mfcc_train.append(mfcc)
        chroma_train.append(chroma)
        label = 0 # Happy
        train_targets.append(label)
    elif 221 <= ind <= 368:
        sc_train.append(sc)
        sb_train.append(sb)
        mfcc_train.append(mfcc)
        chroma_train.append(chroma)
        label = 1 # Sad
        train_targets.append(label)
    elif 380 <= ind <= 537:
        sc_train.append(sc)
        sb_train.append(sb)
        mfcc_train.append(mfcc)
        chroma_train.append(chroma)
        label = 2 # Calm
        train_targets.append(label)
    elif 549 <= ind <= 718:
        sc_train.append(sc)
        sb_train.append(sb)
        mfcc_train.append(mfcc)
        chroma_train.append(chroma)
        label = 3 # Hype
        train_targets.append(label)

sc_net = MyNet(input_dim = sc_train[0].shape[0])
sb_net = MyNet(input_dim = sb_train[0].shape[0])
mfcc_net = MyNet(input_dim = mfcc_train[0].shape[0])
chroma_net = MyNet(input_dim = chroma_train[0].shape[0])

sc_net.apply(weights_init)
sb_net.apply(weights_init)
mfcc_net.apply(weights_init)
chroma_net.apply(weights_init)

sc_data = MyDataset(sc_train, train_targets)
sb_data = MyDataset(sb_train, train_targets)
mfcc_data = MyDataset(mfcc_train, train_targets)
chroma_data = MyDataset(chroma_train, train_targets)

sc_test = MyDataset(sc_test, test_targets)
sb_test = MyDataset(sb_test, test_targets)
mfcc_test = MyDataset(mfcc_test, test_targets)
chroma_test = MyDataset(chroma_test, test_targets)

batch_size = 10

num_epochs = 100

save_dir = "weights"

learning_rate = 0.0001

momentum = 0.9

train_loader = DataLoader(
    ConcatDataset(sc_data, sb_data, mfcc_data, chroma_data),
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    ConcatDataset(sc_test, sb_test, mfcc_test, chroma_test),
    batch_size=batch_size,
    shuffle=True
)

optim_sc = optim.Adam(sc_net.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optim_sb = optim.Adam(sb_net.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optim_mfcc = optim.Adam(mfcc_net.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optim_chroma = optim.Adam(chroma_net.parameters(), lr=learning_rate, betas=(0.5, 0.999))

loss_traj_sc = []
loss_traj_sb = []
loss_traj_mfcc = []
loss_traj_chroma = []

criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    print("Epoch #" + str(epoch+1) + "/" + str(num_epochs))
    
    for phase in ['train', 'test']:
        if phase == 'train':
            sc_net.train()
            sb_net.train()
            mfcc_net.train()
            chroma_net.train()
        else:
            sc_net.eval()
            sb_net.eval()
            mfcc_net.eval()
            chroma_net.eval()

        running_sc_loss = 0.0
        running_sc_corrects = 0

        running_sb_loss = 0.0
        running_sb_corrects = 0

        running_mfcc_loss = 0.0
        running_mfcc_corrects = 0

        running_chroma_loss = 0.0
        running_chroma_corrects = 0

        for i, ((in_sc, label_sc), (in_sb, label_sb), (in_mfcc, label_mfcc), (in_chroma, label_chroma)) in enumerate(train_loader):
            if phase == 'train':
                
                # SC
                sc_net.zero_grad()
                out_sc = sc_net(in_sc.float())
                out_sc_max, sc_preds = torch.max(out_sc, dim=1)
                loss_sc = criterion(out_sc, label_sc)
                loss_sc.backward()
                optim_sc.step()

                # SB
                sb_net.zero_grad()
                out_sb = sb_net(in_sb.float())
                out_sb_max, sb_preds = torch.max(out_sb, dim=1)
                loss_sb = criterion(out_sb, label_sb)
                loss_sb.backward()
                optim_sb.step()

                # MFCC
                mfcc_net.zero_grad()
                out_mfcc = mfcc_net(in_mfcc.float())
                out_mfcc_max, mfcc_preds = torch.max(out_mfcc, dim=1)
                loss_mfcc = criterion(out_mfcc, label_mfcc)
                loss_mfcc.backward()
                optim_mfcc.step()

                # CHROMA
                chroma_net.zero_grad()
                out_chroma = chroma_net(in_chroma.float())
                out_chroma_max, chroma_preds = torch.max(out_chroma, dim=1)
                loss_chroma = criterion(out_chroma, label_chroma)
                loss_chroma.backward()
                optim_chroma.step()
            else:
                # SC
                sc_net.zero_grad()
                out_sc = sc_net(in_sc.float())
                out_sc_max, sc_preds = torch.max(out_sc, dim=1)
                loss_sc = criterion(out_sc, label_sc)

                # SB
                sb_net.zero_grad()
                out_sb = sb_net(in_sb.float())
                out_sb_max, sb_preds = torch.max(out_sb, dim=1)
                loss_sb = criterion(out_sb, label_sb)

                # MFCC
                mfcc_net.zero_grad()
                out_mfcc = mfcc_net(in_mfcc.float())
                out_mfcc_max, mfcc_preds = torch.max(out_mfcc, dim=1)
                loss_mfcc = criterion(out_mfcc, label_mfcc)

                # CHROMA
                chroma_net.zero_grad()
                out_chroma = chroma_net(in_chroma.float())
                out_chroma_max, chroma_preds = torch.max(out_chroma, dim=1)
                loss_chroma = criterion(out_chroma, label_chroma)
            
            running_sc_loss += loss_sc.item() * in_sc.size(0)
            running_sc_corrects += torch.sum(sc_preds == label_sc)

            running_sb_loss += loss_sb.item() * in_sb.size(0)
            running_sb_corrects += torch.sum(sb_preds == label_sb)

            running_mfcc_loss += loss_mfcc.item() * in_mfcc.size(0)
            running_mfcc_corrects += torch.sum(mfcc_preds == label_mfcc)

            running_chroma_loss += loss_chroma.item() * in_chroma.size(0)
            running_chroma_corrects += torch.sum(chroma_preds == label_chroma)
        
        if phase == 'train':
            num_samples = 729-40
        else:
            num_batches = 40
        
        epoch_total_loss = (running_sc_loss+running_sb_loss+running_mfcc_loss+running_chroma_loss) / num_samples
        avg_epoch_acc = (running_chroma_corrects.double()+running_mfcc_corrects.double()+running_sb_corrects.double()+running_sc_corrects.double()) / num_samples / 4

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_total_loss, avg_epoch_acc))

torch.save(sc_net.state_dict(), "sc_net_model.pt")
torch.save(sb_net.state_dict(), "sb_net_model.pt")
torch.save(mfcc_net.state_dict(), "mfcc_net_model.pt")
torch.save(chroma_net.state_dict(), "chroma_net_model.pt")