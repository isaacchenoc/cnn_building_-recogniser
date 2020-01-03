# import library
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# Hyper Parameters
EPOCH_CNN = 0           # num of epoch for CNN. now it is set to 0 as the CNN has been trained         
LR = 0.01              # learning rate
BS = 128                # batch size

train_dataset = tv.datasets.ImageFolder(
    root='png-ZuBuD/train',
    transform=tv.transforms.ToTensor()
)
trainLoader = Data.DataLoader(
    train_dataset,
    batch_size=BS,
    num_workers=0,
    shuffle=True
)
test_dataset = tv.datasets.ImageFolder(
    root='png-ZuBuD/test',
    transform=tv.transforms.ToTensor()
)
testLoader = Data.DataLoader(
    test_dataset,
    batch_size=16,
    num_workers=0,
    shuffle=True
)

# Define my convolutional network 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()             # input: 1 x 120 x 120
        self.conv1 = nn.Conv2d(3, 8, 2, 2)     # output: 16 x 120 x 120
        self.pool  = nn.MaxPool2d(2, 2)          # output: 16 x 60 x 60
        self.conv2 = nn.Conv2d(8, 16, 2, 2)   # output: 32 x 60 x 60
        self.fc1   = nn.Linear(16*30*30, 1005) 
        self.fc2   = nn.Linear(1005, 201)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*30*30)
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


cnn = CNN()
cnn = torch.load('checkpoint12.pth') # load the saved CNN
cnn.eval()

if torch.cuda.is_available(): 
    cnn = cnn.cuda()              # use cuda to help

optimizer = torch.optim.SGD(cnn.parameters(), lr=LR, momentum=0.9)   # optimize all cnn parameters
criterion = nn.CrossEntropyLoss()

correct = 0
total = 0
for epoch in range(EPOCH_CNN): # loop over the dataset multiple times
    for step, (data, target) in enumerate(trainLoader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()     # Use cuda to help 

        # wrap them in Variable
        data, target = Variable(data), Variable(target)
        
        # Bp training
        optimizer.zero_grad() 
        outputs = cnn(data)
        loss = criterion(outputs, target)
        loss.backward()       
        optimizer.step()
        
        # validation
        _, predicted = torch.max(outputs.data, 1)
        total += data.data.size()[0]
        correct += (predicted == target.data).sum()

        if step % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTraining Accuracy: {:.2f}%'.format(
                epoch, step * len(data), len(trainLoader.dataset),
                100. * step / len(trainLoader), loss.item(), correct * 1.0 * 100/ total))    

# Test the performan of CNN using test data
test_loss = 0
test_correct = 0
with torch.no_grad():
    for data, target in testLoader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        output = cnn(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.max(1, keepdim=True)[1]
        test_correct += pred.eq(target.view_as(pred)).sum().item()

        
    test_loss /= len(testLoader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
            .format(test_loss, test_correct, len(testLoader.dataset),
                    100. * test_correct / len(testLoader.dataset)))

plt.ioff()
# Save CNN
#torch.save(cnn, 'checkpoint100.pth')