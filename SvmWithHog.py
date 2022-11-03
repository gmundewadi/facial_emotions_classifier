# imports 
!pip install mlxtend

import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib

import glob
import os
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from skimage.feature import hog
from skimage.io import imread
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import cv2


emotions = ['anger', 'sad', 'happy', 'neutral']

## unzip image files
for emotion in emotions:
    !unzip f'./{emotion}.zip' -d './'




## extract HOG features
samples = []
labels = []

for index, emotion in enumerate(emotions):
  print(f"extracting HOG features for {emotion}")
  # i=0
  for filename in glob.glob(os.path.join(f'./{emotion}', '*'))[:1000]:
    img = cv2.imread(filename, 0)
    hist =  hog(img, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2), block_norm='L1', transform_sqrt=False, feature_vector=True)
    samples.append(hist)
    labels.append(index)
    # i+=1
    # if i%500 == 0:
    #   print(i)
      

samples = np.float32(samples)
hog_features = np.array(samples)
labels = np.array(labels)
labels.reshape(4000,1)


## PCA and dimensionality reduction
scalar = StandardScaler()
df = pd.DataFrame(hog_features)
df_scaled = pd.DataFrame(scalar.fit_transform(df), columns=df.columns)

pca = PCA()
df_pca = pd.DataFrame(pca.fit_transform(df_scaled))

pca = PCA(n_components=1000) # estimate only 1000 PCs
df_new = pca.fit_transform(df_scaled) # project the original data into the PCA space



## training 
df_with_labels = np.append(df_new, labels, axis = 1)
clf = svm.SVC()

np.random.shuffle(df_labels)

percentage = 80 # train-test split
partition = int(len(hog_features)*percentage/100)


x_train, x_test = df_labels[:partition,:-1],  df_labels[partition:,:-1]
y_train, y_test = df_labels[:partition,-1:].ravel() , df_labels[partition:,-1:].ravel()

clf.fit(x_train,y_train)


# testing
y_pred = clf.predict(x_test)

print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))  