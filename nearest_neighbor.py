import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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

k = 40

pred_labels = []
pred_sc = []
pred_sb = []
pred_mfcc = []
pred_chroma = []

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

    out_sc = stats.mode(sc_labels)[0][0]
    pred_sc.append(out_sc)  
    out_sb = stats.mode(sb_labels)[0][0]
    pred_sb.append(out_sb)
    out_mfcc = stats.mode(mfcc_labels)[0][0]
    pred_mfcc.append(out_mfcc)
    out_chroma = stats.mode(chroma_labels)[0][0]
    pred_chroma.append(out_chroma)

sc_correct = np.sum(np.array(pred_sc) == np.array(test_targets))
sb_correct = np.sum(np.array(pred_sb) == np.array(test_targets))
mfcc_correct = np.sum(np.array(pred_mfcc) == np.array(test_targets)) 
chroma_correct = np.sum(np.array(pred_chroma) == np.array(test_targets))
num_correct = np.sum(np.array(pred_labels) == np.array(test_targets))

print(sc_correct/len(test_targets))
print(sb_correct/len(test_targets))
print(mfcc_correct/len(test_targets))
print(chroma_correct/len(test_targets))
print(num_correct/len(test_targets))

total = confusion_matrix(np.array(test_targets), np.array(pred_labels), labels=[0, 1, 2, 3])
plt.imshow(total, interpolation='nearest')
tick_marks = np.arange(4)
plt.xticks(tick_marks, ["happy", "sad", "calm", "hype"], rotation=45)
plt.yticks(tick_marks, ["happy", "sad", "calm", "hype"])
plt.title('Confusion Matrix Total')
plt.savefig('kneighbors_conf.png')
plt.show()

sc = confusion_matrix(np.array(test_targets), np.array(pred_sc), labels=[0, 1, 2, 3])
plt.imshow(sc, interpolation='nearest')
tick_marks = np.arange(4)
plt.xticks(tick_marks, ["happy", "sad", "calm", "hype"], rotation=45)
plt.yticks(tick_marks, ["happy", "sad", "calm", "hype"])
plt.title('Confusion Matrix SC')
plt.savefig('sc_kneighbors_conf.png')
plt.show() 

sb = confusion_matrix(np.array(test_targets), np.array(pred_sb), labels=[0, 1, 2, 3])
plt.imshow(sb, interpolation='nearest')
tick_marks = np.arange(4)
plt.xticks(tick_marks, ["happy", "sad", "calm", "hype"], rotation=45)
plt.yticks(tick_marks, ["happy", "sad", "calm", "hype"])
plt.title('Confusion Matrix SB')
plt.savefig('sb_kneighbors_conf.png')
plt.show()

mfcc = confusion_matrix(np.array(test_targets), np.array(pred_mfcc), labels=[0, 1, 2, 3])
plt.imshow(mfcc, interpolation='nearest')
tick_marks = np.arange(4)
plt.xticks(tick_marks, ["happy", "sad", "calm", "hype"], rotation=45)
plt.yticks(tick_marks, ["happy", "sad", "calm", "hype"])
plt.title('Confusion Matrix MFCC')
plt.savefig('MFCC_kneighbors_conf.png')
plt.show()

chroma = confusion_matrix(np.array(test_targets), np.array(pred_chroma), labels=[0, 1, 2, 3])
plt.imshow(chroma, interpolation='nearest')
tick_marks = np.arange(4)
plt.xticks(tick_marks, ["happy", "sad", "calm", "hype"], rotation=45)
plt.yticks(tick_marks, ["happy", "sad", "calm", "hype"])
plt.title('Confusion Matrix Chroma')
plt.savefig('chroma_kneighbors_conf.png')
plt.show() 