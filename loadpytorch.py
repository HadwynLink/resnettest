import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.models as models
import os
from skimage import io
import pandas as pd
import myResnet
import myDataAug
import myDataset
import onnx
from onnx2pytorch import ConvertModel
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torchvision

def imshow(image):
    #image = image / 2 + 0.5     # unnormalize
    plt.imshow(np.transpose(image, (1, 0, 2)))
    plt.show()

class SPARKDataset(DataLoader):
    def __init__(self, csv_file, root, transform=None):
        self.root_dir = root
        self.satInfo = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.satInfo)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir,
                                self.satInfo.iloc[idx, 0])
        image = io.imread(img_name)
        label = self.satInfo.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)

        return image, label

#Define training data.
training_data = SPARKDataset(
    csv_file="data/SPARK/newtrain.csv",
    root="data/SPARK/train+val",
    transform=ToTensor(),
)

#Define validation data.
val_data = SPARKDataset(
    csv_file="data/SPARK/newval.csv",
    root="data/SPARK/train+val",
    transform=ToTensor(),
)

#Define test data.
test_data = SPARKDataset(
    csv_file="data/SPARK/test_ground_truth.csv",
    root="data/SPARK/test",
    transform=ToTensor(),
)

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(device)

model = myResnet.ResNet(myResnet.ResBlock, [3,4,6,3]).to(device)
model.load_state_dict(torch.load('modelSave.pth', weights_only=True))
model.eval()

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        i = 0
        for X, y in dataloader:
            #transforms = myDataAug.transforms
            #X=transforms(X)
            i += 1
            X, y = X.to(device), y.to(device)
            pred = model(X)
            #print(f'Predicted: "{pred}", Actual: "{y}"')
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss()

test(DataLoader(myDataset.test_data), model, loss_fn)