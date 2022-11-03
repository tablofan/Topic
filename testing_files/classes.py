import torch
import matplotlib.pyplot as plt
import torchvision.datasets
from torchvision import datasets, transforms
from PIL import Image
import os
import natsort
import glob
from typing import List, Tuple
from torchvision.io import read_image
import torchvision.transforms.functional as func
import torch.nn as nn
import torch.nn.functional as F


class Dataset(torch.utils.data.Dataset):
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.img_paths = [path for path in glob.glob(main_dir+"*.jpg", recursive=True)]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        return self.preprocess(index)

    def preprocess(self, index) -> Tuple:
        if index < 0:
            index = 0
        if index >= len(self):
            index = len(self)-1
        image = read_image(self.img_paths[index])
        if self.transform:
            image = self.transform(image)
        image = func.resize(image, [64,64])
        return (image,index)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




