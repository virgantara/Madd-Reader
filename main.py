import numpy as np
import pandas as pd
import os
import glob
from python_speech_features import mfcc, logfbank
from scipy.io import wavfile
from features import extract_features
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model import getHmmModel
import itertools
import pickle
from utils import format_vector
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

input_folder = 'data'

# test_file = '/content/drive/MyDrive/Ayat1/Mad_100_/Mad Iwadl/MadIwadl-(1).wav'
test_file = os.path.join(input_folder,'Mad Layn/MadLayn-(1).wav')

sampling_freq, audio = wavfile.read(test_file)
mfcc_features = mfcc(audio, sampling_freq, nfft=2048)
filterbank_features = logfbank(audio, sampling_freq, nfft=2048)

print ('\nMFCC:\nNumber of windows =', mfcc_features.shape[0])
print ('Length of each feature =', mfcc_features.shape[1])
print ('\nFilter bank: \nNumber of windows =', filterbank_features.shape[0])
print ('Length of each feature =', filterbank_features.shape[1])

df = extract_features(input_folder)

# print(df.head())

# from collections import Counter
# label_cnt = Counter(df.label)

# plt.figure(figsize=(16,8))
# plt.bar(label_cnt.keys(), label_cnt.values())
# plt.title("Dataset labels distribuition")
# plt.show()


label_col = "label"

x_train, x_test, y_train, y_test = train_test_split(df.index, 
                                                      df[label_col].values, 
                                                      test_size=0.2, 
                                                      random_state=109,
                                                      stratify=df[label_col].values)
  
# ambil train dan test dataframe
train_df = df.iloc[x_train]
test_df = df.iloc[x_test]


num_mfcc_features = mfcc_features.shape[1]

def train(num_mfcc_features, label_col, train_df):

	hmm_models = dict()

	model_name = "GaussianHMM"
	feature_col = "mfcc"
	n_components = 10
	cov_type = "diag"
	n_iter = 1000

	# Train HMM perulangan dari df.data suara mad
	mad_sounds = train_df[label_col].unique()

	for mad_sound in mad_sounds:
	    print(mad_sound)
	    mad_sound_df = train_df.loc[train_df[label_col] == mad_sound]

	    model = getHmmModel(model_name, n_components, cov_type, n_iter)
	    model.n_features = num_mfcc_features


	    x_train = format_vector(mad_sound_df[feature_col].values)

	    
	    np.seterr(all='ignore')
	    model.fit(x_train)

	    hmm_models[mad_sound] = model

	full_path = "hmm_models.pickle"
	with open(full_path, 'wb') as handle:
	    pickle.dump(hmm_models, handle, protocol=pickle.HIGHEST_PROTOCOL)

def test(x_test: np.array, hmm_model:dict):
	#predict madnya
	mad_sounds = test_df["label"].unique()

	max_score = -9999999999999999999
	pred_label = None
	pred_probabilities = []

	# evaluasi HMM per Mad untuk predict mad
	for mad_sound in mad_sounds :
		model = hmm_models[mad_sound]
		score = model.score(x_test)
		pred_probabilities.append(score)

		if score > max_score :
	  		max_score = score
	  		pred_label = mad_sound
	
	return np.array([pred_label, pred_probabilities])

## TRAINING
# uncomment to train
# train(num_mfcc_features, label_col, train_df)

##  TESTING
with open("hmm_models.pickle", 'rb') as handle :
 	hmm_models = pickle.load(handle)

# medapatkan prediksi untuk test_df
test_df["y_pred"] = test_df.apply(lambda row: test(row["mfcc"], hmm_models), axis=1)

print(test_df.head())

def plot_confusion_matrix(cm, target_names, normalize, title='Confusion matrix', cmap=plt.cm.Blues):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


y_pred = test_df["y_pred"].values
y_pred = np.asarray([row[0] for row in y_pred])
y_pred_probabilities = np.asarray([row[1] for row in y_pred])
y_true = test_df["label"].values

print(y_pred.shape)
print(y_pred_probabilities.shape)

target_names = test_df["label"].unique()
print(classification_report(y_true, y_pred, target_names=target_names))

cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)

classes = test_df["label"].unique()
plot_confusion_matrix(cm, target_names=classes, normalize=True, title='Normalized confusion matrix')
