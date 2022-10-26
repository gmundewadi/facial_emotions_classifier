import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

# ensure current macOS version at least 12.3+
assert torch.backends.mps.is_available()
# ensure pyTorch installation was built with MPS activated on mac
assert torch.backends.mps.is_built()


#TODO: double-check with paper about transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Load dataset and perform transformations
train_dataset = datasets.ImageFolder('./dataset_pytorch/train', transform = transform)
test_dataset = datasets.ImageFolder('./dataset_pytorch/test', transform = transform)


# Dataloader iteratble returns batches of images and corresponding labels
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=265, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=265, shuffle=True)

#TODO: Data augementation steps goes here

AlexNet_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
AlexNet_model.eval()


#Updating the second classifier
# AlexNet_model.classifier[4] = nn.Linear(4096,1024)

#Updating the third and the last classifier that is the output layer of the network. Make sure to have 10 output nodes if we are going to get 10 class labels through our model.
AlexNet_model.classifier[6] = nn.Linear(1024,4)

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer(SGD)
optimizer = optim.SGD(AlexNet_model.parameters(), lr=0.001, momentum=0.9)

numEpochs = 2 #TODO: run for 50 epochs

for epoch in range(numEpochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0], data[1]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = AlexNet_model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training of AlexNet')
