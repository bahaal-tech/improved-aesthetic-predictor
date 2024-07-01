import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"       # in case you are using a multi GPU workstation, choose your GPU here
import tqdm
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from torchmetrics import Accuracy, F1Score, Precision, Recall


# define your neural net here:

class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Initialize metrics
        self.accuracy = Accuracy()
        self.f1_score = F1Score(num_classes=1, average='weighted')
        self.precision = Precision(num_classes=1, average='weighted')
        self.recall = Recall(num_classes=1, average='weighted')

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)

        # Compute metrics
        acc = self.accuracy(x_hat, y.int())
        f1 = self.f1_score(x_hat, y.int())
        prec = self.precision(x_hat, y.int())
        rec = self.recall(x_hat, y.int())

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_prec', prec, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_rec', rec, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss, 'acc': acc, 'f1': f1, 'prec': prec, 'rec': rec}

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)

        # Compute metrics
        acc = self.accuracy(x_hat, y.int())
        f1 = self.f1_score(x_hat, y.int())
        prec = self.precision(x_hat, y.int())
        rec = self.recall(x_hat, y.int())

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_prec', prec, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_rec', rec, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss, 'acc': acc, 'f1': f1, 'prec': prec, 'rec': rec}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# Load the training data
x = np.load("/mnt/spirit/ava_x.npy")
y = np.load("/mnt/spirit/ava_y.npy")

val_percentage = 0.05  # 5% of the training data will be used for validation
train_border = int(x.shape[0] * (1 - val_percentage))

train_tensor_x = torch.Tensor(x[:train_border])  # transform to torch tensor
train_tensor_y = torch.Tensor(y[:train_border])

train_dataset = TensorDataset(train_tensor_x, train_tensor_y)  # create your dataset
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=16)  # create your dataloader

val_tensor_x = torch.Tensor(x[train_border:])  # transform to torch tensor
val_tensor_y = torch.Tensor(y[train_border:])

val_dataset = TensorDataset(val_tensor_x, val_tensor_y)  # create your dataset
val_loader = DataLoader(val_dataset, batch_size=512, num_workers=16)  # create your dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MLP(768).to(device)  # CLIP embedding dim is 768 for CLIP ViT L 14
optimizer = torch.optim.Adam(model.parameters())

# choose the loss you want to optimize for
criterion = nn.MSELoss()
criterion2 = nn.L1Loss()

epochs = 50

model.train()
best_loss = 999
save_name = "linear_predictor_L14_MSE.pth"

for epoch in range(epochs):
    train_losses = []
    train_accs = []
    train_f1s = []
    train_precs = []
    train_recs = []

    for batch_num, input_data in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = input_data
        x = x.to(device).float()
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        acc = model.accuracy(output, y.int()).item()
        f1 = model.f1_score(output, y.int()).item()
        prec = model.precision(output, y.int()).item()
        rec = model.recall(output, y.int()).item()

        train_losses.append(loss.item())
        train_accs.append(acc)
        train_f1s.append(f1)
        train_precs.append(prec)
        train_recs.append(rec)

        if batch_num % 1000 == 0:
            print(
                f'\tEpoch {epoch} | Batch {batch_num} | Loss {loss.item():.2f} | Acc {acc:.2f} | F1 {f1:.2f} | Prec {prec:.2f} | Rec {rec:.2f}')

    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_train_acc = sum(train_accs) / len(train_accs)
    avg_train_f1 = sum(train_f1s) / len(train_f1s)
    avg_train_prec = sum(train_precs) / len(train_precs)
    avg_train_rec = sum(train_recs) / len(train_recs)

    print(
        f'Epoch {epoch} | Train Loss {avg_train_loss:.2f} | Train Acc {avg_train_acc:.2f} | Train F1 {avg_train_f1:.2f} | Train Prec {avg_train_prec:.2f} | Train Rec {avg_train_rec:.2f}')

    val_losses = []
    val_accs = []
    val_f1s = []
    val_precs = []
    val_recs = []

    for batch_num, input_data in enumerate(val_loader):
        x, y = input_data
        x = x.to(device).float()
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        acc = model.accuracy(output, y.int()).item()
        f1 = model.f1_score(output, y.int()).item()
        prec = model.precision(output, y.int()).item()
        rec = model.recall(output, y.int()).item()

        val_losses.append(loss.item())
        val_accs.append(acc)
        val_f1s.append(f1)
        val_precs.append(prec)
        val_recs.append(rec)

        if batch_num % 1000 == 0:
            print(
                f'\tValidation - Epoch {epoch} | Batch {batch_num} | MSE Loss {loss.item():.2f} | Acc {acc:.2f} | F1 {f1:.2f} | Prec {prec:.2f} | Rec {rec:.2f}')

    avg_val_loss = sum(val_losses) / len(val_losses)
    avg_val_acc = sum(val_accs) / len(val_accs)
    avg_val_f1 = sum(val_f1s) / len(val_f1s)
    avg_val_prec = sum(val_precs) / len(val_precs)
    avg_val_rec = sum(val_recs) / len(val_recs)

    print(
        f'Validation - Epoch {epoch} | MSE Loss {avg_val_loss:.2f} | Acc {avg_val_acc:.2f} | F1 {avg_val_f1:.2f} | Prec {avg_val_prec:.2f} | Rec {avg_val_rec:.2f}')

    if avg_val_loss < best_loss:
        print("Best Val MSE Loss so far. Saving model")
        best_loss = avg_val_loss
        torch.save(model.state_dict(), save_name)

torch.save(model.state_dict(), save_name)

print(f'Best Validation MSE Loss: {best_loss:.2f}')
print("Training done")

# inference test with dummy samples from the val set, sanity check
print("Inference test with dummy samples from the val set, sanity check")
model.eval()
output = model(val_tensor_x[:5].to(device))
print(output.size())
print(output)
