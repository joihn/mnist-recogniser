#!/usr/bin/env python
# coding: utf-8

# # Linear Classification
# 
#  Implement Linear Classification using pytorch. This consists of having fully connected layers connected one after the other and ReLu activation functions between them.
#  
#  Build a neural network with a minimun of 2 layers in order to do classification.

# In[ ]:


import torch
import torch.nn.functional as F
from torch import optim

from torch import nn
from torchvision import datasets, transforms
import numpy as np
import torch.utils.data as utils
import time

from torch.utils.data.sampler import SubsetRandomSampler

torch.manual_seed(1)    # reproducible


#%%
import os
# os.makedirs("input/MNIST/processed", exist_ok=True) # or try this
print(os.listdir("dataset/MNIST/processed"))
#%%
import torch
from torchvision import datasets, transforms

bs = 64 # batch size in every epoch

# trainning set
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root = 'dataset', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=bs, shuffle=True) # shuffle set to True to have the data reshuffled at every epoch.

# test set
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root = 'dataset', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # why using that?
    ])),
    batch_size=bs*2, shuffle=True) # the validation set does not need backpropagation and thus takes less memory,


#%%
from lenet5_like import LeNet5_like

model = LeNet5_like()
print(model)

#%%
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
loss_f = nn.CrossEntropyLoss(reduction = 'mean')

#%%
def train(train_loader, model, optimizer, log_interval, epoch, criterion):
    model.train() # Sets the module in training mode.
    for batch_idx, (inputs, labels) in enumerate(train_loader, 0): # get the inputs. Start from index 0.

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        if batch_idx % log_interval == (log_interval-1):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def test(test_loader, model, criterion):
    model.eval() # Sets the module in evaluation mode.
    test_loss = 0 # loss compute by criterion
    correct = 0 # for computing accurate

    # `with` allows you to ensure that a resource is "cleaned up"
    # when the code that uses it finishes running, even if exceptions are thrown.
    with torch.no_grad(): # It will reduce memory consumption for computations that would otherwise have requires_grad=True.
        for inputs, labels in test_loader:
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item() # sum up batch loss
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader) # average on batches

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


#%%
# Constants
epochs = 5 # how many epochs to train for
log_interval = 200 # how many batches to wait before logging training status
criterion = loss_f

for epoch in range(1, epochs + 1):
    train(train_loader, model, optimizer, log_interval, epoch, criterion)
    test(test_loader, model, criterion)
#%%
torch.save(model.state_dict(), r"C:\Users\maxim\Google Drive\Epfl\MA4\Img analysis\project\mnist recogniser\saved_models\mod1.pkl")



