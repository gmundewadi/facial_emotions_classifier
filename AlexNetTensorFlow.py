import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


# Load data using Keras utility
batch_size = 265
img_height = 224
img_width = 224
data_dir = './dataset/'


train_dataset = tf.keras.utils.image_dataset_from_directory(
  './dataset/train/',
  seed=123, # shuffling and transformations 
  image_size=(img_height, img_width),
  batch_size=batch_size)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
  './dataset/test/',
  seed=123, # shuffling and transformations 
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Output training class names
train_class_names = train_dataset.class_names
print(train_class_names)

# Use buffered prefetching to load images from disck w/out I/O bottlenecks
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)


# AlexNet CNN architecture
# source: https://www.kaggle.com/code/vortexkol/alexnet-cnn-architecture-on-tensorflow-beginner

model=keras.models.Sequential([
    keras.layers.Conv2D(filters=128, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(img_height,img_width,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3)),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(1024,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1024,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4,activation='softmax')  # softmax layer has 4 outputs
])


# Compile AlexNet
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.optimizers.SGD(lr=0.001),
    metrics=['accuracy']    
)
model.summary()


history=model.fit(
    train_dataset,
    epochs=1, # paper sets 50 epochs
    validation_data=validation_dataset,
    validation_freq=1
)

# ['loss', 'accuracy', 'val_loss', 'val_accuracy']
model.history.history.keys()


f,ax=plt.subplots(2,1,figsize=(10,10)) 

#Assigning the first subplot to graph training loss and validation loss
ax[0].plot(model.history.history['loss'],color='b',label='Training Loss')
ax[0].plot(model.history.history['val_loss'],color='r',label='Validation Loss')

plt.legend()

#Plotting the training accuracy and validation accuracy
ax[1].plot(model.history.history['accuracy'],color='b',label='Training  Accuracy')
ax[1].plot(model.history.history['val_accuracy'],color='r',label='Validation Accuracy')

plt.legend()

plt.savefig('./graphs/AlexNetTF.png')

print('Accuracy Score = ',np.max(history.history['val_accuracy']))
