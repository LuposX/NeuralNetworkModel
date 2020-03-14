import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms, datasets
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning import loggers

from PIL import Image


class CNN(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.test_correct_counter = 0
        self.test_total_counter = 0

        self.val_correct_counter = 0
        self.val_total_counter = 0

        self.l1 = nn.Linear(28 * 28, 28 * 28 * 5)
        self.l2 = nn.Linear(28 * 28 * 5, 28 * 28)
        self.l3 = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)

        return x

    def cross_entropy_loss(self, predicted_label, label):
        return F.cross_entropy(predicted_label, label)

    def training_step(self, batch, batch_idx):
        x, y = batch

        predicted_label = self.forward(x)
        loss = self.cross_entropy_loss(predicted_label, y)

        logs = {"train_loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        predicted = self.forward(x)
        loss = self.cross_entropy_loss(predicted, y)

        self.val_correct_counter += int((torch.argmax(predicted, 1).flatten() == y).sum())
        self.val_total_counter += y.size(0)

        logs = {"val_loss": loss}
        return {"val_loss": loss, "logs": logs}


    def validation_epoch_end(self, outputs):
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]

        avg_acc = 100 * self.val_correct_counter / self.val_total_counter

        self.val_correct_counter = 0
        self.val_total_counter = 0

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_acc': avg_acc, 'val_loss': avg_loss}
        return {'avg_val_acc': avg_acc, 'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hat = self.forward(x)

        loss = self.cross_entropy_loss(y_hat, y)

        self.test_correct_counter += int((torch.argmax(y_hat, 1).flatten() == y).sum())
        self.test_total_counter += y.size(0)

        return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        avg_acc = 100 * self.test_correct_counter / self.test_total_counter

        self.test_correct_counter = 0
        self.test_total_counter = 0

        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_test_acc': avg_acc, 'test_loss': avg_loss}

        return {"avg_test_loss": avg_loss, "avg_test_acc": avg_acc, "log: ": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def prepare_data(self):
        compose = transforms.Compose([
            transforms.ToTensor()
        ])

        self.mnist_train = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=compose
        )

        self.mnist_test = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=compose
        )

        self.mnist_train, self.mnist_val = torch.utils.data.random_split(self.mnist_train, [55000, 5000])

    def train_dataloader(self):
        mnist_train_loader = torch.utils.data.DataLoader(self.mnist_train,
                                                         batch_size=128 * 4,
                                                         num_workers=1,
                                                         shuffle=True)

        return mnist_train_loader

    def val_dataloader(self):
        mnist_val_loader = torch.utils.data.DataLoader(self.mnist_val,
                                                         batch_size=128 * 4,
                                                         num_workers=1,
                                                         shuffle=True)

        return mnist_val_loader

    def test_dataloader(self):
        mnist_test_loader = torch.utils.data.DataLoader(self.mnist_test,
                                                       batch_size=128 * 4,
                                                       num_workers=1,
                                                       shuffle=True)

        return mnist_test_loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


if __name__ == "__main__":
    # net = CNN.load_from_checkpoint("tb_logs/NN_08_03_20/linear_nn_for_mnist/checkpoints/epoch=4.ckpt")

    net = CNN()

    trainer = pl.Trainer()
    trainer.fit(net)