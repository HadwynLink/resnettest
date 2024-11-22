import torch

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import os
import pandas as pd
from skimage import io

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