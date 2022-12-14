"""
MobileNetV2TL.py
Transfer learning using MobileNetV2 to predict facial emotions. 
Utilizes Metal Performance Shaders for quick training/validation on Apple Silicon
Author: Gautam Mundewadi
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights

# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
            
transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load dataset and perform transformations (same transforms for train & test)
train_dataset = datasets.ImageFolder('../dataset/train', transform = transform)
test_dataset = datasets.ImageFolder('../dataset/test', transform = transform)

# Dataloader iteratble returns batches of images and corresponding labels
batchSize = 32
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batchSize, shuffle=True)

train_dataset_size = len(trainloader)
test_dataset_size = len(testloader)

mps_device = torch.device('mps')

mobileNetV2_model = models.mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT)
mobileNetV2_model.eval()

#Updating the third and the last classifier that is the output layer of the network. Make sure to have
#  4 output nodes if we are going to get 4 class labels through our model.
mobileNetV2_model.classifier[1] = nn.Linear(mobileNetV2_model.last_channel, 4)

mobileNetV2_model.to(mps_device)

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer(SGD)
optimizer = optim.SGD(mobileNetV2_model.parameters(), lr=0.001, momentum=0.9)

# learning rate decreases step-wise by a factor of .1 ~every 10K iterations
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

nepochs = 10

epochs = list(range(1, nepochs+1))
train_loss_trend, train_acc_trend, test_loss_trend, test_acc_trend = [], [], [], []

for epoch in range(nepochs): 
    
    running_loss = 0.0
    running_accuracy = 0.0

    for i, data in enumerate(trainloader, 0):
        # load input and labels into gpu
        inputs, labels = data[0], data[1]
        inputs = inputs.to(mps_device)
        labels = labels.to(mps_device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # make predictions & calc accuracy
        outputs = mobileNetV2_model(inputs)
        _, preds = torch.max(outputs, 1)
        running_accuracy += float(torch.sum(preds == labels)) / batchSize 

        # forward + backward + optimize
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss_this_epoch = running_loss / train_dataset_size
    train_acc_this_epoch = running_accuracy / train_dataset_size

    train_loss_trend.append(train_loss_this_epoch)
    train_acc_trend.append(train_acc_this_epoch)

    running_loss = 0.0
    running_accuracy = 0.0

    for i, data in enumerate(testloader, 0):
        # load input and labels into gpu
        inputs, labels = data[0], data[1]
        inputs = inputs.to(mps_device)
        labels = labels.to(mps_device)

        # make predictions & calc accuracy 
        outputs = mobileNetV2_model(inputs)
        _, preds = torch.max(outputs, 1)
        running_accuracy += float(torch.sum(preds == labels)) / batchSize

        # compute running loss
        loss = criterion(outputs, labels)
        running_loss += loss.item()

    test_loss_this_epoch = running_loss / test_dataset_size
    test_acc_this_epoch = running_accuracy / test_dataset_size
    
    test_loss_trend.append(test_loss_this_epoch)
    test_acc_trend.append(test_acc_this_epoch)

    print(f'Epoch {epoch + 1} Training Loss: {train_loss_this_epoch:.4f} Training Acc: {train_acc_this_epoch:.4f} Test Loss: {test_loss_this_epoch:.4f} Test Acc: {test_acc_this_epoch:.4f}')\
    

f,ax=plt.subplots(2,1,figsize=(10,10)) 

#Assigning the first subplot to graph training loss and test loss
ax[0].plot(epochs, train_loss_trend,color='b',label='Training Loss')
ax[0].plot(epochs, test_loss_trend,color='r',label='Test Loss')
ax[0].set(xlabel='Epoch', ylabel = 'Loss')
ax[0].legend()

#Plotting the training accuracy and test accuracy
ax[1].plot(epochs,train_acc_trend,color='b',label='Training Accuracy')
ax[1].plot(epochs,test_acc_trend,color='r',label='Test Accuracy')
ax[1].set(xlabel='Epoch', ylabel = 'Accuracy')
ax[1].legend()

plt.savefig('../graphs/MobileNetV2_TL_1126.png')
torch.save(mobileNetV2_model, '../models/MobileNetV2_1126')

print('Accuracy Score = ', np.max(test_acc_trend))
