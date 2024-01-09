import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torchmetrics.classification import MultilabelAccuracy
import pytorch_lightning as pl

from .senet import SENet
from .config import ImgTrainConfig
from .utils import MultiMSE

class ImgRegressor(pl.LightningModule):
    def __init__(self, output_dim: int, feature_ext_model: SENet, cfg: ImgTrainConfig):
        super().__init__()
        self.cfg = cfg
        self.feature_ext_model = feature_ext_model.train()
        if cfg.train_final_layer_only:
            for param in self.feature_ext_model.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p=cfg.dropout)
        self.fc = nn.Linear(self.feature_ext_model.feature_n, output_dim)

        self.loss_func = nn.MSELoss()

        self.acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = MultilabelAccuracy(num_labels=output_dim, average=None)

        self.r2score = torchmetrics.R2Score(output_dim)
        self.val_r2score = torchmetrics.R2Score(output_dim)

        self.val_multi_mse = MultiMSE()

    def forward(self, x):
        x = self.feature_ext_model(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), lr=self.cfg.lr,
            momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
            step_size=10, gamma=self.cfg.gamma
        )
        return [optimizer, ], [scheduler, ]

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
        # これ瞬間のやつ出すだけで epoch でまとめてくれてないっぽい?

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
