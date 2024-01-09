import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torchmetrics.classification import MultilabelAccuracy
from pytorch_pretrained_vit import ViT
import pytorch_lightning as pl
import numpy as np
import albumentations

from .config import ViTImgTrainConfig
from .utils import MultiMSE

INPUT_IMG_SIZE = 384

train_trans = albumentations.Compose([
    albumentations.Resize(INPUT_IMG_SIZE, INPUT_IMG_SIZE),
    albumentations.Blur(p=0.1),
    albumentations.GaussNoise(p=0.1),
    albumentations.ImageCompression(p=0.3),
    albumentations.RandomBrightnessContrast(p=0.3),
    albumentations.CoarseDropout(p=0.7),
    albumentations.Normalize(),
])

test_trans = albumentations.Compose([
    albumentations.Resize(INPUT_IMG_SIZE, INPUT_IMG_SIZE),
    albumentations.Normalize(),
])

def train_transforms(img) -> torch.Tensor:
    img = np.array(img, dtype=np.uint8)
    img = train_trans(image=img)['image']
    img = img.astype(np.float32)
    img = img.transpose(2, 0, 1)  # C x H x W
    img = torch.from_numpy(img).float()
    return img

def test_transforms(img) -> torch.Tensor:
    img = np.array(img, dtype=np.uint8)
    img = test_trans(image=img)['image']
    img = img.astype(np.float32)
    img = img.transpose(2, 0, 1)  # C x H x W
    img = torch.from_numpy(img).float()
    return img

class ViTRegressor(pl.LightningModule):
    def __init__(self, output_dim: int, cfg: ViTImgTrainConfig):
        super().__init__()
        self.cfg = cfg

        self.vit = ViT('B_32_imagenet1k', pretrained=True)
        self.vit.train()
        for param in self.vit.parameters():
            param.requires_grad = False

        self.vit.fc = nn.Linear(self.vit.fc.in_features, output_dim)

        self.loss_func = nn.MSELoss()

        self.acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = MultilabelAccuracy(num_labels=output_dim, average=None)

        self.r2score = torchmetrics.R2Score(output_dim)
        self.val_r2score = torchmetrics.R2Score(output_dim)

        self.val_multi_mse = MultiMSE()

    def forward(self, x):
        x = self.vit(x)
        return x

    def configure_optimizers(self):
        return optim.Adam(self.vit.parameters(), lr=self.cfg.lr)

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
