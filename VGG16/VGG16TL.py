"""
VGG16TL.py
Transfer learning using VGG16 to predict facial emotions. 
Utilizes Metal Performance Shaders for quick training/validation on Apple Silicon
Author: Gautam Mundewadi
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torchvision.models import VGG16_Weights

# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
            
train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Load dataset and perform transformations
train_dataset = datasets.ImageFolder('../dataset/train', transform = train_transform)
test_dataset = datasets.ImageFolder('../dataset/test', transform = test_transform)

# Dataloader iteratble returns batches of images and corresponding labels
batchSize = 256 #TODO: consider other batch sizes
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batchSize, shuffle=True)

num_training_points = len(train_dataset) 
test_dataset_size = len(testloader) # number of test datapoints per batch


mps_device = torch.device('mps')

VGG16_model = models.vgg16(weights = VGG16_Weights.DEFAULT)
VGG16_model.eval()

#Updating the third and the last classifier that is the output layer of the network. Make sure to have 4 output nodes if we are going to get 4 class labels through our model.
VGG16_model.classifier[6] = nn.Linear(4096,4)
VGG16_model.to(mps_device)

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer(SGD)
optimizer = optim.SGD(VGG16_model.parameters(), lr=0.001, momentum=0.9)

# learning rate decreases step-wise by a factor of .1 ~every 10K iterations
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

niters = 200 # niters 

iters = list(range(1, niters+1))
train_loss_trend, train_acc_trend, test_loss_trend, test_acc_trend = [], [], [], []

for iter in range(niters): 
    
    running_loss = 0.0
    running_accuracy = 0.0

    subset = torch.utils.data.Subset(train_dataset, [random.randint(0, num_training_points-1)]) # select 1 datapoint for SGD
    trainloader_stochastic = torch.utils.data.DataLoader(subset, batch_size=1, num_workers=0, shuffle=False)
    
    for i, data in enumerate(trainloader_stochastic, 0):
        # load input and labels into gpu
        inputs, labels = data[0], data[1]

        inputs = inputs.to(mps_device)
        labels = labels.to(mps_device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # make predictions & calc accuracy
        outputs = VGG16_model(inputs)
        _, preds = torch.max(outputs, 1)
        running_accuracy += float(torch.sum(preds == labels)) / batchSize 

        # forward + backward + optimize
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss_this_iter = running_loss
    train_acc_this_iter = running_accuracy 

    train_loss_trend.append(train_loss_this_iter)
    train_acc_trend.append(train_acc_this_iter)

    running_loss = 0.0
    running_accuracy = 0.0

    for i, data in enumerate(testloader, 0):
        # load input and labels into gpu
        inputs, labels = data[0], data[1]
        inputs = inputs.to(mps_device)
        labels = labels.to(mps_device)

        # make predictions & calc accuracy 
        outputs = VGG16_model(inputs)
        _, preds = torch.max(outputs, 1)
        running_accuracy += float(torch.sum(preds == labels)) / batchSize

        # compute running loss
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        print(f'Evaluation step: {i} / {test_dataset_size}')

    test_loss_this_iter = running_loss / test_dataset_size
    test_acc_this_iter = running_accuracy / test_dataset_size
    
    test_loss_trend.append(test_loss_this_iter)
    test_acc_trend.append(test_acc_this_iter)

    print(f'Iteration {iter + 1} Training Loss: {train_loss_this_iter:.4f} Training Acc: {train_acc_this_iter:.4f} Test Loss: {test_loss_this_iter:.4f} Test Acc: {test_acc_this_iter:.4f}')\
    


f,ax=plt.subplots(2,1,figsize=(10,10)) 

#Assigning the first subplot to graph training loss and test loss
ax[0].plot(iters, train_loss_trend,color='b',label='Training Loss')
ax[0].plot(iters, test_loss_trend,color='r',label='Test Loss')
ax[0].set(xlabel='Iteration', ylabel = 'Loss')
ax[0].legend()

#Plotting the training accuracy and test accuracy
ax[1].plot(iters,train_acc_trend,color='b',label='Training Accuracy')
ax[1].plot(iters,test_acc_trend,color='r',label='Test Accuracy')
ax[1].set(xlabel='Iteration', ylabel = 'Accuracy')
ax[1].legend()

plt.savefig('../graphs/VGG16_TL.png')
torch.save(VGG16_model, '../models/VGG16_TL')

print('Accuracy Score = ', np.max(test_acc_trend))
