import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torchmetrics.classification import MultilabelAccuracy
import pytorch_lightning as pl

from .config import AudioTrainConfig
from .utils import MultiMSE

class AudioModel(nn.Module):
    def __init__(self, feature_dim: int, output_dim: int, two_layer: bool):
        super(AudioModel, self).__init__()

        if two_layer:
            middle_dim = 128
            self.layer = nn.Sequential(
                nn.Linear(feature_dim, middle_dim),
                nn.ReLU(),
                nn.Linear(middle_dim, output_dim),
            )
        else:
            self.layer = nn.Sequential(
                nn.Linear(feature_dim, output_dim),
            )

    def forward(self, x):
        x = self.layer(x)
        return x

class AudioRegressor(pl.LightningModule):
    def __init__(self, feature_dim: int, output_dim: int, cfg: AudioTrainConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.model = AudioModel(feature_dim, output_dim, cfg.two_layer)

        self.loss_func = nn.MSELoss()

        self.acc = torchmetrics.Accuracy(task="binary")
        self.r2score = torchmetrics.R2Score(output_dim)

        self.val_acc = MultilabelAccuracy(num_labels=output_dim, average=None)
        self.val_r2score = torchmetrics.R2Score(output_dim)
        self.val_multi_mse = MultiMSE()

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), lr=self.cfg.lr,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
            step_size=1, gamma=self.cfg.gamma
        )
        return [optimizer, ], [scheduler, ]

    def training_step(self, batch, batch_idx):
        x, labels = batch

        x = self.forward(x)

        loss = self.loss_func(x, labels)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        # """
        r2 = self.r2score(x, labels)
        self.log("train_R2", r2, prog_bar=True, sync_dist=True)

        x = torch.where(x >= 0.0, 1.0, 0.0)
        labels = torch.where(labels >= 0.0, 1.0, 0.0)
        self.log("train_acc", self.acc(x, labels), prog_bar=True, sync_dist=True)

        return loss

    def training_epoch_end(self, outputs) -> None:
        self.log("train_epoch_R2", self.r2score.compute(), on_epoch=True, sync_dist=True)
        self.r2score.reset()
        self.log("train_epoch_acc", self.acc.compute(), prog_bar=True, sync_dist=True)
        self.acc.reset()

    def validation_step(self, batch, batch_idx):
        x, labels = batch

        x = self.forward(x)

        loss = self.loss_func(x, labels)
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True)

        self.val_multi_mse.update(x, labels)

        # """
        r2 = self.val_r2score(x, labels)
        self.log("valid_R2", r2, prog_bar=True, sync_dist=True)

        x = torch.where(x >= 0.0, 1.0, 0.0)
        labels = torch.where(labels >= 0.0, 1.0, 0.0)
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
