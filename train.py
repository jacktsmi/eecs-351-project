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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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
    sc = np.load("train_sc_moresamps/" + i + ".npy")
    sb = np.load("train_sb_moresamps/" + i + ".npy")
    mfcc = np.load("train_mfcc_moresamps/" + i + ".npy")
    chroma = np.load("train_chroma_moresamps/" + i + ".npy")
    mfcc = mfcc.reshape((mfcc.shape[0]*mfcc.shape[1]))
    chroma = chroma.reshape((chroma.shape[0]*chroma.shape[1]))
    if 0 < ind < 12: # or 368 < ind < 380 or 537 < ind < 549 or 718 < ind < 730:
        sc_test.append(sc)
        sb_test.append(sb)
        mfcc_test.append(mfcc)
        chroma_test.append(chroma)
        label = 0 # Happy
        test_targets.append(label)
    elif 220 < ind < 232:
        sc_test.append(sc)
        sb_test.append(sb)
        mfcc_test.append(mfcc)
        chroma_test.append(chroma)
        label = 1 # Sad
        test_targets.append(label)
    elif 379 < ind < 391:
        sc_test.append(sc)
        sb_test.append(sb)
        mfcc_test.append(mfcc)
        chroma_test.append(chroma)
        label = 2 # Calm
        test_targets.append(label)
    elif 548 < ind < 560:
        sc_test.append(sc)
        sb_test.append(sb)
        mfcc_test.append(mfcc)
        chroma_test.append(chroma)
        label = 3 # Hype
        test_targets.append(label)
    elif 12 <= ind <= 159:
        sc_train.append(sc)
        sb_train.append(sb)
        mfcc_train.append(mfcc)
        chroma_train.append(chroma)
        label = 0 # Happy
        train_targets.append(label)
    elif 232 <= ind <= 379:
        sc_train.append(sc)
        sb_train.append(sb)
        mfcc_train.append(mfcc)
        chroma_train.append(chroma)
        label = 1 # Sad
        train_targets.append(label)
    elif 391 <= ind <= 391+147:
        sc_train.append(sc)
        sb_train.append(sb)
        mfcc_train.append(mfcc)
        chroma_train.append(chroma)
        label = 2 # Calm
        train_targets.append(label)
    elif 560 <= ind <= 560+147:
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

# sc_data = MyDataset(sc_train, train_targets)
# sb_data = MyDataset(sb_train, train_targets)
mfcc_data = MyDataset(mfcc_train, train_targets)
#chroma_data = MyDataset(chroma_train, train_targets)

#sc_test = MyDataset(sc_test, test_targets)
#sb_test = MyDataset(sb_test, test_targets)
mfcc_test = MyDataset(mfcc_test, test_targets)
#chroma_test = MyDataset(chroma_test, test_targets)

batch_size = 40

num_epochs = 5

save_dir = "weights"

learning_rate = 0.0001

train_loader = DataLoader(
    mfcc_data,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    mfcc_test,
    batch_size=batch_size,
    shuffle=True
)

#optim_sc = optim.Adam(sc_net.parameters(), lr=learning_rate, betas=(0.5, 0.999))
#optim_sb = optim.Adam(sb_net.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optim_mfcc = optim.Adam(mfcc_net.parameters(), lr=learning_rate, betas=(0.5, 0.999))
#optim_chroma = optim.Adam(chroma_net.parameters(), lr=learning_rate, betas=(0.5, 0.999))

loss_traj_sc = []
loss_traj_sb = []
loss_traj_mfcc = []
loss_traj_chroma = []

sc_acc = []
sb_acc = []
mfcc_acc = []
chroma_acc = []

test_acc = []
train_acc = []

num_classes = 4
out_data = torch.zeros((batch_size, num_classes))

criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    print("Epoch #" + str(epoch+1) + "/" + str(num_epochs))
    
    #sc_net.train()
    #sb_net.train()
    mfcc_net.train()
    #chroma_net.train()

    running_mfcc_loss = 0.0
    running_mfcc_loss_test = 0.0
    running_mfcc_corrects = 0
    running_mfcc_corrects_test = 0

    running_corrects = 0
    running_corrects_test = 0

    for i, ((in_mfcc, label_mfcc)) in enumerate(train_loader):            

        # MFCC
        mfcc_net.zero_grad()
        out_mfcc = mfcc_net(in_mfcc.float())
        out_mfcc_max, mfcc_preds = torch.max(out_mfcc, dim=1)
        loss_mfcc = criterion(out_mfcc, label_mfcc)
        loss_mfcc.backward()
        optim_mfcc.step()

        out_max, preds = torch.max(out_mfcc, dim=1)

        running_corrects += torch.sum(preds == label_mfcc) # Doesn't matter which label since they're the same

        running_mfcc_loss += loss_mfcc.item() * in_mfcc.size(0)
        running_mfcc_corrects += torch.sum(mfcc_preds == label_mfcc)

    mfcc_net.eval()

    for i, ((in_mfcc, label_mfcc)) in enumerate(test_loader):  

        # MFCC
        mfcc_net.zero_grad()
        out_mfcc = mfcc_net(in_mfcc.float())
        out_mfcc_max, mfcc_preds = torch.max(out_mfcc, dim=1)
        loss_mfcc = criterion(out_mfcc, label_mfcc)

        out_max, preds = torch.max(out_mfcc, dim=1)

        running_corrects_test += torch.sum(preds == label_mfcc) # Doesn't matter which label since they're the same

        running_mfcc_loss_test += loss_mfcc.item() * in_mfcc.size(0)
        running_mfcc_corrects_test += torch.sum(mfcc_preds == label_mfcc)

        # Compute and save confusion matrix for first batch each epoch
        if i == 0:
            c = confusion_matrix(label_mfcc.detach().numpy(), mfcc_preds, labels=[0, 1, 2, 3])
            plt.imshow(c, interpolation='nearest')
            tick_marks = np.arange(4)
            plt.xticks(tick_marks, ["happy", "sad", "calm", "hype"], rotation=45)
            plt.yticks(tick_marks, ["happy", "sad", "calm", "hype"])
            plt.title('Confusion Matrix'.format(epoch))
            plt.savefig(f'conf_mats/conf_mat_mfcc{epoch}.png')
            plt.show()

    num_samples_train =  148*4
    num_samples_test =  44

    # Total loss from all NN's

    # Calculating average accuracy over the 4 NN's
    #train_epoch_acc = (running_chroma_corrects.double()+running_mfcc_corrects.double()+running_sb_corrects.double()+running_sc_corrects.double()) / num_samples_train / 4
    #test_epoch_acc = (running_chroma_corrects_test+running_mfcc_corrects_test+running_sb_corrects_test+running_sc_corrects_test) / num_samples_test / 4
    train_epoch_acc = (running_mfcc_corrects.double()) / num_samples_train
    test_epoch_acc = (running_mfcc_corrects_test) / num_samples_test

    train_acc.append(train_epoch_acc.item())
    test_acc.append(test_epoch_acc.item())

    # Taking individual accuracies
    mfcc_test_acc = running_mfcc_corrects_test.double() / num_samples_test

    mfcc_acc.append(mfcc_test_acc.item())

    # Calculating accuracies from adding NN outputs before taking max
    epoch_acc_train = running_corrects / num_samples_train
    epoch_acc_test = running_corrects_test / num_samples_test

    print('{} Acc: {:.4f}'.format('train', train_epoch_acc))
    print('{} Acc: {:.4f}'.format('test', test_epoch_acc))
    #print("SC:" + str(sc_test_acc.item()) + " SB:" + str(sb_test_acc.item()) + " MFCC:" + str(mfcc_test_acc.item()) + " Chroma:" + str(chroma_test_acc.item()))

loss_traj_mfcc.append(running_mfcc_loss_test)

torch.save(mfcc_net.state_dict(), "mfcc_net_model2.pt")

fig2, ax2 = plt.subplots()
ax2.plot(test_acc, 'b', label='Test Accuracy')
ax2.plot(train_acc, 'r', label='Train Accuracy')
ax2.set_title('Training and Test Accuracies')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()

fig2.savefig("Accuracies.png")
plt.show()