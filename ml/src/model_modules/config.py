from dataclasses import dataclass


@dataclass
class ImgTrainConfig:
    max_epoch: int
    batch_size: int
    dropout: float
    lr: float
    momentum: float
    gamma: float
    weight_decay: float
    train_final_layer_only: bool

@dataclass
class ViTImgTrainConfig:
    max_epoch: int
    batch_size: int
    lr: float
    train_final_layer_only: bool

@dataclass
class AudioTrainConfig:
    max_epoch: int
    batch_size: int
    lr: float
    momentum: float
    gamma: float
    weight_decay: float
    two_layer: bool
