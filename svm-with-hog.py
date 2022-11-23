import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import matplotlib.pyplot as plt
import cv2

import numpy as np


from random import shuffle 
from tqdm import tqdm 
import warnings
warnings.filterwarnings('ignore')
import os
import pickle
from sklearn import svm
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


train_features_path = "./images-train"
test_features_path = "./images-test"
train_labels_path = "./annotations-train"
test_labels_path = "./annotations-test"

train_size = 12000
test_size = 3000
image_size = 200

#load train and test data
train_features = []
test_features = []

for image in tqdm(os.listdir(train_features_path)[:train_size]): 
        path = os.path.join(train_features_path, image)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        hist =  hog(img, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2), block_norm='L1', transform_sqrt=False, feature_vector=True)
        train_features.append(hist)

for image in tqdm(os.listdir(test_features_path)[:test_size]): 
        path = os.path.join(test_features_path, image)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        hist =  hog(img, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2), block_norm='L1', transform_sqrt=False, feature_vector=True)
        test_features.append(hist)


train_labels = []
test_labels = []

for image in tqdm(os.listdir(train_labels_path)[:test_size]): 
        path = os.path.join(train_labels_path, image)
        value = int(np.load(path))
        train_labels.append(value) # need to remap

for image in tqdm(os.listdir(test_labels_path)[:test_size]): 
        path = os.path.join(test_labels_path, image)
        value = int(np.load(path))
        test_labels.append(value) # need to remap

# remaping anger: 0, happy: 1, neutral: 2, sad: 3

for i in range(len(train_labels)):
    if train_labels[i] == 0:
        train_labels[i] = 2
    elif train_labels[i] == 2:
        train_labels[i] = 3
    elif train_labels[i] == 6:
        train_labels[i] = 0

for i in range(len(test_labels)):
    if test_labels[i] == 0:
        test_labels[i] = 2
    elif test_labels[i] == 2:
        test_labels[i] = 3
    elif test_labels[i] == 6:
        test_labels[i] = 0

number_of_train = train_features.shape[0]
number_of_test = test_features.shape[0]

x_train_flatten = train_features.reshape(number_of_train,train_features.shape[1]*train_features.shape[2])
x_test_flatten = test_features.reshape(number_of_test,test_features.shape[1]*test_features.shape[2])

scalar = StandardScaler()

df = pd.DataFrame(x_train_flatten)
df_scaled = pd.DataFrame(scalar.fit_transform(df), columns=df.columns)

pca = PCA(n_components=1200)
df_pca = pd.DataFrame(pca.fit_transform(df_scaled))

clf = svm.SVC()
clf.fit(df_pca,train_labels)

df_test = pd.DataFrame(x_test_flatten)
df_test_pca = pca.fit_transform(df_test)
clf.score(df_test_pca, test_labels)

# save the model
filename = 'svm-hog.sav'
pickle.dump(clf, open(filename, 'wb'))

# load the model
# loaded_model = pickle.load(open(filename, 'rb'))