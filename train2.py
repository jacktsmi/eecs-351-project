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

train = []
test = []
train_targets = []
test_targets = []

for ind in range(1, 730):
    i = str(ind)
    sc = np.load("train_spectral_centroid/" + i + ".npy")
    sb = np.load("train_spectral_bandwidth/" + i + ".npy")
    mfcc = np.load("train_mfcc/" + i + ".npy")
    chroma = np.load("train_chroma/" + i + ".npy")
    #mfcc = mfcc.reshape((mfcc.shape[0]*mfcc.shape[1]))
    #chroma = chroma.reshape((chroma.shape[0]*chroma.shape[1]))
    data = np.concatenate((sc, sb, mfcc, chroma))
    if 0 < ind < 12: # or 368 < ind < 380 or 537 < ind < 549 or 718 < ind < 730:
        test.append(data)
        label = 0 # Happy
        test_targets.append(label)
    elif 220 < ind < 232:
        test.append(data)
        label = 1 # Sad
        test_targets.append(label)
    elif 379 < ind < 391:
        test.append(data)
        label = 2 # Calm
        test_targets.append(label)
    elif 548 < ind < 560:
        test.append(data)
        label = 3 # Hype
        test_targets.append(label)
    elif 12 <= ind <= 220:
        train.append(data)
        label = 0 # Happy
        train_targets.append(label)
    elif 232 <= ind <= 379:
        train.append(data)
        label = 1 # Sad
        train_targets.append(label)
    elif 391 <= ind <= 548:
        train.append(data)
        label = 2 # Calm
        train_targets.append(label)
    elif 560 <= ind <= 729:
        train.append(data)
        label = 3 # Hype
        train_targets.append(label)

net = MyNet(input_dim = train[0].shape[0])

net.apply(weights_init)

train_data = MyDataset(train, train_targets)

test_data = MyDataset(test, test_targets)

batch_size = 100

num_epochs = 50

save_dir = "weights"

learning_rate = 0.0001

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=True
)

optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.5, 0.999))

loss_traj = []

test_acc = []
train_acc = []

num_classes = 4
out_data = torch.zeros((batch_size, num_classes))

criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    print("Epoch #" + str(epoch+1) + "/" + str(num_epochs))
    
    net.train()

    running_loss_train = 0.0
    running_loss_test = 0.0

    running_corrects_train = 0
    running_corrects_test = 0

    for i, (in_vec, label) in enumerate(train_loader):            
        # SC
        net.zero_grad()
        out = net(in_vec.float())
        out_max, preds = torch.max(out, dim=1)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        running_corrects_train += torch.sum(preds == label) # Doesn't matter which label since they're the same
        running_loss_train += loss.item() * in_vec.size(0)
    
    net.eval()

    for i, (in_vec, label) in enumerate(test_loader):  
        # SC
        net.zero_grad()
        out = net(in_vec.float())
        out_max, preds = torch.max(out, dim=1)
        loss = criterion(out, label)

        running_corrects_test += torch.sum(preds == label)
        running_loss_test += loss.item() * in_vec.size(0)

        # Compute and save confusion matrix for first batch each epoch
        if i == 0:
            c = confusion_matrix(label.detach().numpy(), preds, labels=[0, 1, 2, 3])
            plt.imshow(c, interpolation='nearest')
            tick_marks = np.arange(4)
            plt.xticks(tick_marks, ["happy", "sad", "calm", "hype"], rotation=45)
            plt.yticks(tick_marks, ["happy", "sad", "calm", "hype"])
            plt.title('Confusion Matrix'.format(epoch))
            plt.savefig(f'conf_mats/conf_mat_epoch{epoch}.png')
            plt.show()
    
    num_samples_train =  729-44
    num_samples_test =  44

    # Total loss from all NN's
    epoch_train_loss = running_loss_train / num_samples_train
    epoch_test_loss = running_loss_test / num_samples_test

    # Calculating average accuracy over the 4 NN's
    train_epoch_acc = running_corrects_train.double() / num_samples_train
    test_epoch_acc = running_corrects_test.double() / num_samples_test

    train_acc.append(train_epoch_acc.item())
    test_acc.append(test_epoch_acc.item())

    print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_train_loss, train_epoch_acc))
    print('{} Loss: {:.4f} Acc: {:.4f}'.format('test', epoch_test_loss, test_epoch_acc))
    #print("SC:" + str(sc_test_acc.item()) + " SB:" + str(sb_test_acc.item()) + " MFCC:" + str(mfcc_test_acc.item()) + " Chroma:" + str(chroma_test_acc.item()))

torch.save(net.state_dict(), "net_model3.pt")

fig2, ax2 = plt.subplots()
ax2.plot(test_acc, 'b', label='Test Accuracy')
ax2.plot(train_acc, 'r', label='Train Accuracy')
ax2.legend()

fig2.savefig("Accuracies.png")
plt.show()