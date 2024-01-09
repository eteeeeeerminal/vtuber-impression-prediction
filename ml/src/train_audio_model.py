import pathlib

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
import torch
from torch.utils.data import DataLoader, Subset, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from utils.logging import get_logger
from utils.config import AudioClassifierTrainConfig
from dataset.vtuber_audios_dataset import VTuberAudioDataset
from dataset.utils import split_dataset
from model_modules.utils import LossErrorStopping
from model_modules.audio_pl import AudioRegressor


@hydra.main(version_base=None, config_path="../configs", config_name="audio_regressor")
def main(cfg: AudioClassifierTrainConfig):
    pl.seed_everything(cfg.seed, workers=True)

    logger = get_logger(__name__)

    # config 周りの調整
    cfg.save_dir = pathlib.Path(cfg.save_dir)
    hydra_cfg = HydraConfig.get()
    if hydra_cfg.mode == RunMode.MULTIRUN:
        cfg.save_dir = cfg.save_dir.joinpath(hydra_cfg.sweep.subdir)
        cfg.version = cfg.version + f"-{hydra_cfg.sweep.subdir}"

    # データの構築
    logger.info("Load dataset.")
    dataset = VTuberAudioDataset(cfg.dataset)
    train_dataset, valid_dataset, test_dataset = split_dataset(
        dataset, cfg.valid_ratio, cfg.test_ratio
    )

    # united_dataset = Subset(dataset, train_dataset.indices + valid_dataset.indices)
    # train_dataset, valid_dataset = random_split(united_dataset, [len(train_dataset), len(valid_dataset)])

    train_loader = DataLoader(train_dataset, cfg.train.batch_size, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, cfg.train.batch_size, num_workers=4, persistent_workers=True, pin_memory=True)

    # モデル構築
    logger.info("Init model.")
    # """
    model = AudioRegressor(dataset.get_feature_dim(), dataset.get_label_dim(), cfg.train)

    # checkpoint のセーブの設定
    checkpoint_callback = ModelCheckpoint(
        cfg.save_dir.joinpath("model"), monitor="valid_loss",
        filename="audio-{epoch:02d}-{valid_loss:.4f}",
        save_top_k=1, mode="min",
        save_on_train_epoch_end=True
    )

    error_stopping = LossErrorStopping("train_loss")

    # logger
    logger_csv = CSVLogger(".", version=cfg.version)
    logger_tb = TensorBoardLogger(".", version=logger_csv.version)
    logger.info(f"pl running version: {logger_csv.version}")

    # Trainer の構築と学習の実行
    trainer = pl.Trainer(
        accelerator="gpu", devices=[0, 1], max_epochs=cfg.train.max_epoch,
        logger = [logger_csv, logger_tb],
        callbacks=[checkpoint_callback, error_stopping], check_val_every_n_epoch=None,
        val_check_interval=1000,
        strategy=DDPStrategy(find_unused_parameters=False)
    )
    logger.info(f"start training")
    trainer.fit(model, train_loader, valid_loader)
    logger.info(f"DONE!")
    # """


if __name__ == "__main__":
    main()
