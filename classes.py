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
        print(self.total_imgs)

    def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.total_imgs)

    # def __getitem__(self, index):
    #     # 'Generates one sample of data'
    #     # Select sample
    #     ID = self.list_IDs[index]
    #
    #     # Load data and get label
    #     X = torch.load('data/' + ID + '.pt')
    #     y = self.labels[ID]
    #
    #     return X, y

