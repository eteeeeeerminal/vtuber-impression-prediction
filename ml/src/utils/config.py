from dataclasses import dataclass

from dataset.vtuber_imgs_dataset import ImgDatasetConfig
from dataset.vtuber_audios_dataset import AudioDatasetConfig
from model_modules.config import ImgTrainConfig, ViTImgTrainConfig, AudioTrainConfig

@dataclass
class ImageClassifierTrainConfig:
    seed: int
    save_dir: str
    experiment_id: str
    version: str
    pretrained_path: str
    valid_ratio: float
    test_ratio: float
    dataset: ImgDatasetConfig
    train: ImgTrainConfig

@dataclass
class ViTImgRegressorConfig:
    seed: int
    save_dir: str
    experiment_id: str
    version: str
    valid_ratio: float
    test_ratio: float
    dataset: ImgDatasetConfig
    train: ViTImgTrainConfig

@dataclass
class AudioClassifierTrainConfig:
    seed: int
    save_dir: str
    experiment_id: str
    version: str
    valid_ratio: float
    test_ratio: float
    dataset: AudioDatasetConfig
    train: AudioTrainConfig
