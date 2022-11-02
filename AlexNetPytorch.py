import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torchvision.models import AlexNet_Weights

# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
            
#TODO: double-check with paper about transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Load dataset and perform transformations
train_dataset = datasets.ImageFolder('./dataset/train', transform = transform)
test_dataset = datasets.ImageFolder('./dataset/test', transform = transform)


# Dataloader iteratble returns batches of images and corresponding labels
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=265, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=265, shuffle=True)

#TODO: Data augementation steps goes here

mps_device = torch.device('mps')

AlexNet_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', weights = AlexNet_Weights.DEFAULT)
AlexNet_model.eval()


# Reduces overfitting?
# AlexNet_model.classifier[4] = nn.Linear(4096,1024)
# AlexNet_model.classifier[6] = nn.Linear(1024, 4)

#Updating the third and the last classifier that is the output layer of the network. Make sure to have 4 output nodes if we are going to get 4 class labels through our model.
AlexNet_model.classifier[6] = nn.Linear(4096,4)
AlexNet_model.to(mps_device)

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer(SGD)
optimizer = optim.SGD(AlexNet_model.parameters(), lr=0.001, momentum=0.9)

# learning rate decreases step-wise by a factor of .1 ~every 10K iterations
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

numEpochs = 2 #TODO: run for 50 epochs

print("Beginning Training ...")

for epoch in range(numEpochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0], data[1]
        inputs = inputs.to(mps_device)
        labels = labels.to(mps_device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = AlexNet_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
        

print('Finished Training of AlexNet')
torch.save(AlexNet_model.state_dict(), "./models/AlexNet_TL.pt")
