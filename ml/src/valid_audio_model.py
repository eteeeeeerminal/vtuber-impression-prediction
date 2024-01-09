import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from torch.utils.data import DataLoader, Subset, random_split
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger

from utils.logging import get_logger
from dataset.vtuber_audios_dataset import VTuberAudioDataset
from dataset.utils import split_dataset
from utils.config import AudioClassifierTrainConfig
from model_modules.senet import senet50, load_state_dict
from model_modules.const_model import AverageModel
from model_modules.audio_pl import AudioRegressor


@hydra.main(version_base=None, config_path="../configs", config_name="audio_regressor")
def main(cfg: AudioClassifierTrainConfig):
    pl.seed_everything(cfg.seed, workers=True)

    logger = get_logger(__name__)

    hydra_cfg = HydraConfig.get()
    if hydra_cfg.mode == RunMode.MULTIRUN:
        cfg.version = cfg.version + f"-{hydra_cfg.sweep.subdir}"

    # データの構築
    logger.info("Load dataset.")
    dataset = VTuberAudioDataset(cfg.dataset)
    train_dataset, valid_dataset, test_dataset = split_dataset(
        dataset, cfg.valid_ratio, cfg.test_ratio
    )

    # united_dataset = Subset(dataset, train_dataset.indices + valid_dataset.indices)
    # train_dataset, valid_dataset = random_split(united_dataset, [len(train_dataset), len(valid_dataset)])

    logger.info(f"train:valid:test={len(train_dataset)}:{len(valid_dataset)}:{len(test_dataset)}")

    train_loader = DataLoader(train_dataset, cfg.train.batch_size, num_workers=4, persistent_workers=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, cfg.train.batch_size, num_workers=4, persistent_workers=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, cfg.train.batch_size, num_workers=4, persistent_workers=True, pin_memory=True)

    # モデル構築
    # """
    logger.info("Init model.")
    cfg.train.two_layer = False
    model = AudioRegressor(dataset.get_feature_dim(), dataset.get_label_dim(), cfg.train)

    average_model = AverageModel(dataset.get_label_dim())
    average_model.train_average_model(train_loader)
    # checkpoint_path = "outputs/2023-01-05T12-33-20-audio-2e69f291/23/model/audio-epoch=01-valid_loss=0.0987.ckpt"
    checkpoint_path = "outputs/2023-01-06T09-04-40-audio-e71b1f34/17/model/audio-epoch=01-valid_loss=0.1007.ckpt"
    print(model.state_dict())
    loaded = torch.load(checkpoint_path)
    model.load_state_dict(loaded['state_dict'])
    print(model.state_dict())

    logger.info(f"loaded checkpoint {checkpoint_path}")

    logger_csv = CSVLogger(".", version=cfg.version)
    logger.info(f"pl running version: {logger_csv.version}")

    # Trainer の構築と学習の実行
    trainer = pl.Trainer(
        accelerator="gpu", devices=[1], max_epochs=cfg.train.max_epoch, num_nodes=1,
        logger = logger_csv,
        check_val_every_n_epoch=None,
        val_check_interval=500,
        strategy=DDPStrategy(find_unused_parameters=False)
    )

    logger.info(f"start checkpoint model validation")
    logger.info(f"TEST ------")
    trainer.validate(model, test_loader)
    logger.info(f"VALID ------")
    trainer.validate(model, valid_loader)
    logger.info(f"TRAIN ------")
    trainer.validate(model, train_loader) 
    logger.info(f"DONE!")

    logger.info(f"start average model validation")
    logger.info(f"TEST ------")
    trainer.validate(average_model, test_loader) 
    logger.info(f"VALID ------")
    trainer.validate(average_model, valid_loader)
    logger.info(f"TRAIN ------")
    trainer.validate(average_model, train_loader)
    logger.info(f"DONE!")
    # """


if __name__ == "__main__":
    main()


