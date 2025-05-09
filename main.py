import torch
import os
import torch.nn as nn
# import torchvision
# import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

# W&B
import wandb

# Dotenv
from dotenv import load_dotenv

# PyTorch Lightning imports
import lightning as pl
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
load_dotenv()

# Hyperparams
BATCH_SIZE=32
NUM_WORKERS=4
HIDDEN_SIZE=500
NUM_CLASSES=10
NUM_EPOCHS=10
LR=0.001
INPUT_SIZE=784 # 28x28

# Replace nn.Module with pl.LightningModule
# pl.LightningModule is a superset of nn.Module
class NN(pl.LightningModule): 
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(NN, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.num_classes = num_classes
        self.validation_step_outputs = []
        
        self.loss_fn = F.cross_entropy
        self.accuracy = Accuracy(task="multiclass", num_classes=10)

        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

        # W&B Logging
        # Saves to self.hparams which are auto-logged by W&B
        self.save_hyperparameters()

    # Returns raw logits (non-normalized) with no sigmoid or softmax applied
    def forward(self, x):
        return self.l2(self.relu(self.l1(x)))
    
    def _get_preds_loss_accuracy(self, batch):
        """Helper function, train/valid/test steps are similar"""
        images, labels = batch
        images = images.reshape(-1, 28*28)
        logits = self(images)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss_fn(logits, labels)
        acc = self.accuracy(preds, labels)
        return preds, loss, acc

    # Definition of single batch forward pass over training
    def training_step(self, batch, batch_index):
        """Returns loss from a single batch"""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # Logging
        self.log("train_loss", loss, logger=True)
        self.log("train_accuracy", acc, logger=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        """Logging metrics"""
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # Logging
        self.log("val_loss", loss, logger=True)
        self.log("val_accuracy", acc, logger=True)
        return preds

    def train_dataloader(self):
        """Provides the configured dataloader object for training"""
        train_dataset = MNIST(root="./data", train=True, download=True, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
        return train_loader
    
    def val_dataloader(self):
        """Provides the configured dataloader object for validation"""
        test_dataset = MNIST(root="./data", train=False, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
        return test_loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = LR)

if __name__ == '__main__':

    # Configure logger
    logger = TensorBoardLogger("tb_logs", name="PT-Lightning-MNIST")

    # Define model and train
    model = NN(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
    trainer = Trainer(max_epochs=NUM_EPOCHS, logger=logger)
    trainer.fit(model)

    # TODO: Look into implementing Lighting DDP for distributed training across mutliple machines 