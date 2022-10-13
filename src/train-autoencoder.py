# https://linuxtut.com/en/a062d0c245c8f4836399/

import os

import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28)
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    dataset = MNIST('.data', download=True, transform=transforms.ToTensor())
    train, val = random_split(dataset, [55000, 5000])

    autoencoder = LitAutoEncoder()
    trainer = pl.Trainer(max_epochs=1)  # , gpus=1)
    trainer.fit(
        autoencoder,
        DataLoader(train, batch_size=256, num_workers=4),
        DataLoader(val, batch_size=256, num_workers=4),
    )

    # torchscript
    autoencoder = LitAutoEncoder()
    torch.jit.save(autoencoder.to_torchscript(), "model.pt")
    print("training_finished")
