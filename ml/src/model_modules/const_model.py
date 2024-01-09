import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics
from torchmetrics.classification import MultilabelAccuracy
import pytorch_lightning as pl

from .utils import MultiMSE

class AverageModel(pl.LightningModule):
    """データセットの平均を常に出力"""
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.average = torch.zeros(output_dim)

        self.loss_func = nn.MSELoss()

        self.acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = MultilabelAccuracy(num_labels=output_dim, average=None)

        self.r2score = torchmetrics.R2Score(output_dim)
        self.val_r2score = torchmetrics.R2Score(output_dim)

        self.val_multi_mse = MultiMSE()

    def train_average_model(self, dataloader: DataLoader):
        data_n = 0
        self.average = torch.zeros(self.output_dim)
        for _, labels in dataloader:
            data_n += labels.shape[0]
            self.average += torch.sum(labels, 0)
        self.average /= data_n
        print(self.average)

    def forward(self, x):
        batch_n = x.shape[0]
        output = self.average.repeat(batch_n, 1)
        output = output.to(x.device)
        return output

    def training_step(self, batch, batch_idx):
        imgs, labels = batch

        x = self.forward(imgs)

        loss = self.loss_func(x, labels)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        # """
        r2 = self.r2score(x, labels)
        self.log("train_R2", r2, prog_bar=True, sync_dist=True)

        x = torch.where(x >= 0, 1.0, 0.0)
        labels = torch.where(labels >= 0, 1.0, 0.0)
        self.log("train_acc", self.acc(x, labels), prog_bar=True, sync_dist=True)

        return loss

    def training_epoch_end(self, outputs) -> None:
        self.log("train_epoch_R2", self.r2score.compute(), prog_bar=True, sync_dist=True)
        self.r2score.reset()
        self.log("train_epoch_acc", self.acc.compute(), prog_bar=True, sync_dist=True)
        self.acc.reset()

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch

        x = self.forward(imgs)

        loss = self.loss_func(x, labels)
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True)

        self.val_multi_mse.update(x, labels)

        # """
        r2 = self.val_r2score(x, labels)
        self.log("valid_R2", r2, prog_bar=True, sync_dist=True)

        x = torch.where(x >= 0, 1.0, 0.0)
        labels = torch.where(labels >= 0, 1.0, 0.0)
        scores = self.val_acc(x, labels)
        self.log("valid_acc", scores.mean(), prog_bar=True, sync_dist=True)

        return loss

    def validation_epoch_end(self, outputs) -> None:
        self.log("valid_epoch_R2", self.val_r2score.compute(), sync_dist=True)
        self.val_r2score.reset()

        scores = self.val_acc.compute()
        self.log("valid_epoch_acc", scores.mean(), prog_bar=True, sync_dist=True)
        for i, score in enumerate(scores):
            self.log(f"valid_epoch_acc_{i}", score, prog_bar=False, sync_dist=True)
        self.val_acc.reset()

        losses = self.val_multi_mse.compute()
        for i, loss in enumerate(losses):
            self.log(f"valid_epoch_loss_{i}", loss, prog_bar=False, sync_dist=True)
        self.val_multi_mse.reset()

