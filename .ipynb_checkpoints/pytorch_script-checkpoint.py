import torch
import matplotlib.pyplot as plt
# from torchvision import datasets, transforms
from classes import Dataset, Net
# import numpy as np
# from PIL import Image
# import cv2
# import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
# from torchvision.datasets import MNIST
import torch.optim as optim


if __name__ == "__main__":
    # CUDA
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    # torch.backends.cudnn.benchmark = True

    # Parameters
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 6}
    max_epochs = 1

    # Datasets
    dataset = Dataset("data/Raphael/")
    test_image = dataset[0]
    print(test_image)
    plt.imshow(test_image[0].permute(1,2,0))
    plt.show()

    # loader = DataLoader(data,batch_size=12, num_workers=2)
    # for i, batch in enumerate(loader):
    #     print(i, batch)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=16,shuffle=True, num_workers=2)
    criterion = nn.CrossEntropyLoss()
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')