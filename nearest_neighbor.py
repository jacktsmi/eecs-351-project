import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

train_sc = np.zeros((250, 148*4))
train_sb = np.zeros((250, 148*4))
train_mfcc = np.zeros((5976, 148*4))
train_chroma = np.zeros((25*12, 148*4))

test_sc = np.zeros((250, 44))
test_sb = np.zeros((250, 44))
test_mfcc = np.zeros((5976, 44))
test_chroma = np.zeros((25*12, 44))

train_targets = []
test_targets = []

accuracies = []

test_i = 0
train_i = 0

for ind in range(1, 730):
    i = str(ind)
    sc = np.load("train_spectral_centroid/" + i + ".npy")
    sb = np.load("train_spectral_bandwidth/" + i + ".npy")
    mfcc = np.load("train_mfcc/" + i + ".npy")
    chroma = np.load("train_chroma/" + i + ".npy")

    if 0 < ind < 12: # or 368 < ind < 380 or 537 < ind < 549 or 718 < ind < 730:
        test_sc[:,test_i] = sc
        test_sb[:,test_i] = sb
        test_mfcc[:,test_i] = mfcc
        test_chroma[:,test_i] = chroma
        test_i += 1
        label = 0 # Happy
        test_targets.append(label)
    elif 220 < ind < 232:
        test_sc[:,test_i] = sc
        test_sb[:,test_i] = sb
        test_mfcc[:,test_i] = mfcc
        test_chroma[:,test_i] = chroma
        test_i += 1
        label = 1 # Sad
        test_targets.append(label)
    elif 379 < ind < 391:
        test_sc[:,test_i] = sc
        test_sb[:,test_i] = sb
        test_mfcc[:,test_i] = mfcc
        test_chroma[:,test_i] = chroma
        test_i += 1
        label = 2 # Calm
        test_targets.append(label)
    elif 548 < ind < 560:
        test_sc[:,test_i] = sc
        test_sb[:,test_i] = sb
        test_mfcc[:,test_i] = mfcc
        test_chroma[:,test_i] = chroma
        test_i += 1
        label = 3 # Hype
        test_targets.append(label)
    elif 12 <= ind <= 12+147:
        train_sc[:,train_i] = sc
        train_sb[:,train_i] = sb
        train_mfcc[:,train_i] = mfcc
        train_chroma[:,train_i] = chroma
        train_i += 1
        label = 0 # Happy
        train_targets.append(label)
    elif 232 <= ind <= 232+147:
        train_sc[:,train_i] = sc
        train_sb[:,train_i] = sb
        train_mfcc[:,train_i] = mfcc
        train_chroma[:,train_i] = chroma
        train_i += 1
        label = 1 # Sad
        train_targets.append(label)
    elif 391 <= ind <= 391+147:
        train_sc[:,train_i] = sc
        train_sb[:,train_i] = sb
        train_mfcc[:,train_i] = mfcc
        train_chroma[:,train_i] = chroma
        train_i += 1
        label = 2 # Calm
        train_targets.append(label)
    elif 560 <= ind <= 560+147:
        train_sc[:,train_i] = sc
        train_sb[:,train_i] = sb
        train_mfcc[:,train_i] = mfcc
        train_chroma[:,train_i] = chroma
        train_i += 1
        label = 3 # Hype
        train_targets.append(label)

for k in range(1, 60):

    pred_labels = []

    for j in range(test_sc.shape[1]): 
        sc_diff = train_sc - np.tile(test_sc[:,j], (148*4,1)).T
        sb_diff = train_sb - np.tile(test_sb[:,j], (148*4,1)).T
        mfcc_diff = train_mfcc - np.tile(test_mfcc[:,j], (148*4,1)).T
        chroma_diff = train_chroma - np.tile(test_chroma[:,j], (148*4,1)).T

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
        pred_labels.append(out)
    
       
    num_correct = np.sum(np.array(pred_labels) == np.array(test_targets))
    accuracies.append(num_correct*100/len(test_targets))

plt.plot(range(1, 60), accuracies)
plt.xlabel('k')
plt.ylabel('Accuracies (%)')
plt.savefig('k_neighbors.png')
plt.show()