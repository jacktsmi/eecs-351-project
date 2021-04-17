from scipy import stats
from sklearn import svm
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

train_sc = np.zeros((250, 685))
train_sb = np.zeros((250, 685))
train_mfcc = np.zeros((5976, 685))
train_chroma = np.zeros((25*12, 685))

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
        test_sc[:, test_i] = sc
        test_sb[:, test_i] = sb
        test_mfcc[:, test_i] = mfcc
        test_chroma[:, test_i] = chroma
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
    elif 12 <= ind <= 220:
        train_sc[:,train_i] = sc
        train_sb[:,train_i] = sb
        train_mfcc[:,train_i] = mfcc
        train_chroma[:,train_i] = chroma
        train_i += 1
        label = 0 # Happy
        train_targets.append(label)
    elif 232 <= ind <= 379:
        train_sc[:,train_i] = sc
        train_sb[:,train_i] = sb
        train_mfcc[:,train_i] = mfcc
        train_chroma[:,train_i] = chroma
        train_i += 1
        label = 1 # Sad
        train_targets.append(label)
    elif 391 <= ind <= 548:
        train_sc[:,train_i] = sc
        train_sb[:,train_i] = sb
        train_mfcc[:,train_i] = mfcc
        train_chroma[:,train_i] = chroma
        train_i += 1
        label = 2 # Calm
        train_targets.append(label)
    elif 560 <= ind <= 729:
        train_sc[:,train_i] = sc
        train_sb[:,train_i] = sb
        train_mfcc[:,train_i] = mfcc
        train_chroma[:,train_i] = chroma
        train_i += 1
        label = 3 # Hype
        train_targets.append(label)

clf_sc = svm.SVC(C=1, class_weight='balanced')
clf_sb = svm.SVC(C=1, class_weight='balanced')
clf_mfcc = svm.SVC(C=1, class_weight='balanced')
clf_chroma = svm.SVC(C=1, class_weight='balanced')

clf_sc.fit(train_sc.T, train_targets)
clf_sb.fit(train_sb.T, train_targets)
clf_mfcc.fit(train_mfcc.T, train_targets)
clf_chroma.fit(train_chroma.T, train_targets)

sc_preds = clf_sc.predict(test_sc.T)
sb_preds = clf_sb.predict(test_sb.T)
mfcc_preds = clf_mfcc.predict(test_mfcc.T)
chroma_preds = clf_chroma.predict(test_chroma.T)

c = confusion_matrix(np.array(test_targets), np.array(sc_preds), labels=[0, 1, 2, 3])
plt.imshow(c, interpolation='nearest')
tick_marks = np.arange(4)
plt.xticks(tick_marks, ["happy", "sad", "calm", "hype"], rotation=45)
plt.yticks(tick_marks, ["happy", "sad", "calm", "hype"])
plt.title('Confusion Matrix for Spectral Centroid')
plt.savefig('svm_conf_sc.png')
plt.show()

corrects_sc = np.sum(np.array(sc_preds) == np.array(test_targets))
accuracy_sc = corrects_sc/len(test_targets)
print("Spectral Centroid Accuracy: " + str(accuracy_sc))


c = confusion_matrix(np.array(test_targets), np.array(sb_preds), labels=[0, 1, 2, 3])
plt.imshow(c, interpolation='nearest')
tick_marks = np.arange(4)
plt.xticks(tick_marks, ["happy", "sad", "calm", "hype"], rotation=45)
plt.yticks(tick_marks, ["happy", "sad", "calm", "hype"])
plt.title('Confusion Matrix for Spectral Bandwidth')
plt.savefig('svm_conf_sb.png')
plt.show()

corrects_sb = np.sum(np.array(sb_preds) == np.array(test_targets))
accuracy_sb = corrects_sb/len(test_targets)
print("Spectral Bandwidth Accuracy: " + str(accuracy_sc))


c = confusion_matrix(np.array(test_targets), np.array(mfcc_preds), labels=[0, 1, 2, 3])
plt.imshow(c, interpolation='nearest')
tick_marks = np.arange(4)
plt.xticks(tick_marks, ["happy", "sad", "calm", "hype"], rotation=45)
plt.yticks(tick_marks, ["happy", "sad", "calm", "hype"])
plt.title('Confusion Matrix for MFCC')
plt.savefig('svm_conf_mfcc.png')
plt.show()

corrects_mfcc = np.sum(np.array(mfcc_preds) == np.array(test_targets))
accuracy_mfcc = corrects_mfcc/len(test_targets)
print("MFCC Accuracy: " + str(accuracy_sc))

c = confusion_matrix(np.array(test_targets), np.array(chroma_preds), labels=[0, 1, 2, 3])
plt.imshow(c, interpolation='nearest')
tick_marks = np.arange(4)
plt.xticks(tick_marks, ["happy", "sad", "calm", "hype"], rotation=45)
plt.yticks(tick_marks, ["happy", "sad", "calm", "hype"])
plt.title('Confusion Matrix for Spectral Chroma')
plt.savefig('svm_conf_chroma.png')
plt.show()

corrects_chroma = np.sum(np.array(chroma_preds) == np.array(test_targets))
accuracy_chroma = corrects_chroma/len(test_targets)
print("Spectral Chroma Accuracy: " + str(accuracy_chroma))

total_pred = np.stack((sc_preds, sb_preds, mfcc_preds, chroma_preds), axis=-1)
preds = stats.mode(total_pred, axis=1)[0]
corrects = np.sum(np.array(preds.squeeze()) == np.array(test_targets))

c = confusion_matrix(np.array(test_targets), np.array(preds), labels=[0, 1, 2, 3])
plt.imshow(c, interpolation='nearest')
tick_marks = np.arange(4)
plt.xticks(tick_marks, ["happy", "sad", "calm", "hype"], rotation=45)
plt.yticks(tick_marks, ["happy", "sad", "calm", "hype"])
plt.title('Confusion Matrix')
plt.savefig('svm_conf.png')
plt.show()

accuracy = corrects/len(test_targets)
print("Total Accuracy: " + str(accuracy))


