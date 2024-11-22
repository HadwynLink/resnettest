import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import v2 
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.models as models

import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

import os
import pandas as pd
from skimage import io
import signal
import myResnet
import myDataset
import myDataAug

batch_size = 64
epochs = 5
learning_rate = 1e-3
weight_decay = 0.001
momentum = 0.9

with open('hyperparams.csv', newline='') as csvfile:

    params = csv.reader(csvfile, delimiter=',', quotechar='|')
    i = 0
    for row in params:
        match i:
            case 0:
                batch_size = int(row[1])
            case 1:
                epochs = int(row[1])
            case 2:
                learning_rate = float(row[1])
            case 3:
                weight_decay = float(row[1])
            case 4:
                momentum = float(row[1])
        i += 1

hiscore = 0

training_data = myDataset.training_data
test_data = myDataset.test_data
val_data = myDataset.val_data

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def imshow(img):
    img = img[:3, :, :]
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#2,2,2,2: ResNet 18
#3,4,6,3: ResNet 34
#use bottleneck for ResNet 50+
#3,4,23,3: ResNet 101
#3,8,35,3: ResNet 152
model = myResnet.ResNet(myResnet.ResBlock, [3,4,6,3]).to(device)
model.load_state_dict(torch.load('modelSave.pth', weights_only=True))
print(model)

def handler(signum, frame):
    torch.save(model.state_dict(), "modelSave.pth")
    print("Saved PyTorch Model State to modelSave.pth")
    sys.exit(0)  

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), learning_rate, weight_decay, momentum)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        cor = 0
        #do image transforms here
        #transforms = myDataAug.transforms

        #X = transforms(X)

        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)

        cor += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} % correct: {(cor/batch_size)*100} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            transforms = myDataAug.transforms
            X=transforms(X)
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


signal.signal(signal.SIGINT, handler)
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(val_dataloader, model, loss_fn)
print("Done!")
test(test_dataloader, model, loss_fn)

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

classes = [
    "smart-1",
    "cheops",
    "lisa_pathfinder",
    "debris",
    "proba_3_ocs",
    "proba_3_csc",
    "soho",
    "earth_observation_sat_1",
    "proba_2",
    "xmm_newton",
    "double_star",
]
