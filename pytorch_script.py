import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from classes import Dataset
import numpy as np
from PIL import Image
import cv2
# import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST


if __name__ == "__main__":
    # CUDA
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    # torch.backends.cudnn.benchmark = True

    # Parameters
    # params = {'batch_size': 64,
    #           'shuffle': True,
    #           'num_workers': 6}
    # max_epochs = 100

    # Datasets
    data = Dataset("data/Raphael")
    # imageset = np.empty(shape=len(data))
    # for i,v in enumerate(data.total_imgs):
    #     imageset[i] = np.array(Image.open(f'{data.main_dir}/{v}'))
    # print(imageset)
    # print(imageset.shape)
    test_image = cv2.imread(f'{data.main_dir}/{data.total_imgs[0]}')
    test_image_resized = cv2.resize(test_image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    test_image_normalized = cv2.normalize(test_image_resized, None, alpha=0,beta=200, norm_type=cv2.NORM_MINMAX)
    test_image_rgb_normalized = (test_image_resized - np.min(test_image_resized))/np.ptp(test_image_resized)
    print(test_image.shape)
    print(test_image_resized.shape)
    # print(test_image_resized[:3][:3][:3])
    # print(test_image_normalized[:3][:3][:3])
    print(test_image_rgb_normalized)
    plt.imshow(test_image)
    plt.show()
    plt.imshow(test_image_resized)
    plt.show()
    plt.imshow(test_image_normalized)
    plt.show()
    plt.imshow(test_image_resized)
    plt.show()

    loader = DataLoader(data,batch_size=12, num_workers=2)
    for i, batch in enumerate(loader):
        print(i, batch)