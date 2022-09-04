import torch
import matplotlib.pyplot as plt
import torchvision.datasets
from torchvision import datasets, transforms
from PIL import Image
import os
import natsort


class Dataset(torch.utils.data.Dataset):
    def __init__(self, main_dir):
        # 'Initialization'
        self.main_dir = main_dir
        # self.transform = transform
        self.total_imgs = natsort.natsorted(os.listdir(main_dir))
        self.list_IDs = []

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, index):
        return self.total_imgs[index]

