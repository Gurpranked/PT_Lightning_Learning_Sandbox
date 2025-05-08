import torch
import os
import torch.nn as nn
# import torchvision
# import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# PyTorch Lightning imports
import pytorch_lightning as pl
from pytorch_lightning import Trainer

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
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, max_steps: int):
        super(NN, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.num_classes = num_classes
        self.validation_step_outputs = []
        self.max_steps = max_steps

        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    # Returns raw logits (non-normalized) with no sigmoid or softmax applied
    def forward(self, x):
        return self.l2(self.relu(self.l1(x)))
    
    # Definition of single batch forward pass over training
    def training_step(self, batch, batch_index):
        images, labels = batch
        images = images.reshape(-1, 28*28)
        max_steps = self.max_steps

        y_hat = self(images)
        loss = F.cross_entropy(y_hat, labels)

        tenserboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tenserboard_logs}
    
    # Obtain training dataloader
    def train_dataloader(self):
        train_dataset = MNIST(root="./data", train=True, download=True, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
        return train_loader
    
    def val_dataloader(self):
        test_dataset = MNIST(root="./data", train=False, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
        return test_loader
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28 * 28)

        # Forward pass
        y_hat = self(images)

        loss = F.cross_entropy(y_hat, labels)
        self.validation_step_outputs.append({'val_loss': loss})  

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = LR)

if __name__ == '__main__':
    model = NN(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, max_steps=10)
    trainer = Trainer(max_epochs=NUM_EPOCHS, deterministic=True)
    trainer.fit(model)

    # TODO: Look into implementing Lighting DDP for distributed training across mutliple machines 