# HERE YOUR IMPORTS
import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms, datasets
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from PIL import Image


class CNN(pl.LightningModule):
    def __init__(self, batch_size, lr, ndf):
        super().__init__()
        # Parameters
        self.batch_size = batch_size
        self.lr = lr

        self.val_correct_counter = 0  # for accuracy
        self.val_total_counter = 0  # for accuracy

        # DEFINE HERE ARCHITECTURE
        # ---------------------------

    def forward(self, x):
        # HERE THE FORWARD METHOD
        # ---------------------------

        return x

    def cross_entropy_loss(self, predicted_label, label):
        # EXAMPLE LOSS METHOD
        return F.cross_entropy(predicted_label, label)  # your loss

    def training_step(self, batch, batch_idx):
        # EXAMPLE TRAINING STEP
        x, y = batch  # split up your batch

        predicted = self.forward(x)  # make a prediction with your nn
        loss = self.cross_entropy_loss(predicted, y)  # compute the loss

        logs = {"train_loss": loss}
        return {"loss": loss, "log": logs}  # return the loss

    def validation_step(self, val_batch, batch_idx):
        # EXAMPLE VALIDATION STEP
        x, y = val_batch

        predicted = self.forward(x)  # make a prediction with your nn
        loss = self.cross_entropy_loss(predicted, y)  # compute the loss

        # create a confusion matrix
        comet_logger.experiment.log_confusion_matrix(labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                                                     y_true=torch.eye(10)[y].view(-1, 10),
                                                     y_predicted=predicted
                                                     )

        # increase the counter of correct classified validation examples
        self.val_correct_counter += int((torch.argmax(predicted, 1).flatten() == y).sum())
        self.val_total_counter += y.size(0)  # increase the total counter of validation examples

        logs = {"val_loss": loss}  # for logging

        # return the loss for display and logging
        return {"val_loss": loss, "log": logs}


    def validation_epoch_end(self, outputs):
        # EXAMPLE VALIDATION END

        avg_acc = 100 * self.val_correct_counter / self.val_total_counter  # calculate the accuracy of validation

        # reset the values for calculating accuracy
        self.val_correct_counter = 0
        self.val_total_counter = 0

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()  # calculate the avg validation loss
        logs = {'avg_val_acc': avg_acc, 'val_loss': avg_loss}  # for logging

        # return the loss for display and logging
        return {'avg_val_acc': avg_acc, 'avg_val_loss': avg_loss, 'log': logs}

    def test_step(self, test_batch, batch_idx):
        # TEST VALIDATION END
        x, y = test_batch   # split up your batch
        y_hat = self.forward(x)  # make a prediction with your nn

        loss = self.cross_entropy_loss(y_hat, y)  # calculate the loss

        return {"test_loss": loss}   # return the loss for display

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()   # calculate the avg test loss
        logs = {'avg_test_acc': avg_acc, 'test_loss': avg_loss}  # for logging

        #  return the loss for display and logging
        return {"avg_test_loss": avg_loss, "avg_test_acc": avg_acc, "log: ": tensorboard_logs, 'progress_bar': logs}

    def prepare_data(self):
        # transform our dataset into tensor
        compose = transforms.Compose([
            transforms.ToTensor()
        ])

        # create our dataset
        self.dataset_train = datasets.MNIST(  # for training
            root="data",
            train=True,
            download=True,
            transform=compose
        )

        self.dataset_test = datasets.MNIST(  # for testing
            root="data",
            train=False,
            download=True,
            transform=compose
        )

        self.datasets_train, self.datasets_val = torch.utils.data.random_split(self.mnist_train, [55000, 5000])  # for validation

    def train_dataloader(self):
        mnist_train_loader = torch.utils.data.DataLoader(self.datasets_train,
                                                         batch_size=self.batch_size,
                                                         num_workers=1,
                                                         shuffle=True)

        return mnist_train_loader

    def val_dataloader(self):
        mnist_val_loader = torch.utils.data.DataLoader(self.datasets_val,
                                                         batch_size=self.batch_size,
                                                         num_workers=1,
                                                         shuffle=True)

        return mnist_val_loader

    def test_dataloader(self):
        mnist_test_loader = torch.utils.data.DataLoader(self.mnist_test,
                                                       batch_size=self.batch_size,
                                                       num_workers=1,
                                                       shuffle=True)

        return mnist_test_loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    # net = CNN.load_from_checkpoint("tb_logs/NN_08_03_20/linear_nn_for_mnist/checkpoints/epoch=4.ckpt")

    # Parameters
    experiment_name = "cnn_0"  # The name of your experiment is shown in comet.ml
    code_file_name = "cnn_mnist_classifier_train.py"  # to upload the code to comet.ml
    dataset_name = "MNIST"  # gets logged
    checkpoint_folder = "./" + experiment_name + "_checkpoints/"  # where we save our checkpoints
    tags = ["cnn", "test"]  # tags seen in comet.ml

    # Hyperparameters
    lr = 0.001
    batch_size = 128*4
    ndf = 16

    # Create our loggger
    comet_logger = loggers.CometLogger(
        api_key=os.environ["COMET_KEY"],
        rest_api_key=os.environ["COMET_REST_KEY"],
        project_name="mnist-classifier",
        experiment_name=experiment_name,
    )

    # Neural Network
    net = CNN(batch_size, lr, ndf)

    # Init stuff
    comet_logger.experiment.set_code(open(code_file_name, "r").read(), overwrite=True)
    comet_logger.experiment.set_model_graph(str(net))
    comet_logger.experiment.add_tags(tags=tags)
    comet_logger.experiment.log_dataset_info(name=dataset_name)
    comet_logger.experiment.log_parameter(name="learning_rate", value=lr)
    comet_logger.experiment.log_parameter(name="batch_size", value=batch_size)
    comet_logger.experiment.log_parameter(name="ndf", value=ndf)

    # Training the NN
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_folder, save_top_k=3)
    trainer = pl.Trainer(fast_dev_run=True,
                         checkpoint_callback=checkpoint_callback,
                         val_check_interval=0.5,
                         max_epochs=20,
                         logger=comet_logger)
    trainer.fit(net)

    # log checkpoints
    comet_logger.experiment.log_asset_folder(folder=checkpoint_folder)

