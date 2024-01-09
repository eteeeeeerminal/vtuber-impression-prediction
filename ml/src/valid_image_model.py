import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from torch.utils.data import DataLoader, Subset, random_split
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger

from utils.logging import get_logger
from utils.config import ImageClassifierTrainConfig
from dataset.vtuber_imgs_dataset import VTuberImgDataset, train_transforms, test_transforms
from dataset.utils import split_dataset
from model_modules.senet import senet50, load_state_dict
from model_modules.pl import ImgRegressor
from model_modules.const_model import AverageModel


@hydra.main(version_base=None, config_path="../configs", config_name="image_regressor")
def main(cfg: ImageClassifierTrainConfig):
    pl.seed_everything(cfg.seed, workers=True)

    logger = get_logger(__name__)

    hydra_cfg = HydraConfig.get()
    if hydra_cfg.mode == RunMode.MULTIRUN:
        cfg.version = cfg.version + f"-{hydra_cfg.sweep.subdir}"

    # データの構築
    logger.info("Load dataset.")
    dataset = VTuberImgDataset(cfg.dataset, None)
    train_dataset, valid_dataset, test_dataset = split_dataset(
        dataset, cfg.valid_ratio, cfg.test_ratio,
        train_transform=test_transforms,
        valid_transform=test_transforms,
        test_transform=test_transforms
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
    senet = senet50()
    if cfg.pretrained_path:
        load_state_dict(senet, cfg.pretrained_path)

    average_model = AverageModel(dataset.get_label_dim())
    average_model.train_average_model(train_loader)
    # checkpoint_path = "outputs/2023-01-11T13-09-44-image-637fc1dd/5/model/senet-epoch=00-valid_loss=0.1349.ckpt"
    checkpoint_path = "outputs/2023-01-11T13-09-44-image-637fc1dd/10/model/senet-epoch=00-valid_loss=0.1105.ckpt"
    # checkpoint_path = "outputs/2023-01-09T15-24-39-image-03ea23f3/9/model/senet-epoch=01-valid_loss=0.1101.ckpt"
    cfg.train.dropout = 0.5
    model = ImgRegressor(
        output_dim= dataset.get_label_dim(),
        feature_ext_model=senet,
        cfg=cfg.train
    )
    # model.load_from_checkpoint(
    #     checkpoint_path,
    #     output_dim=dataset.get_label_dim(),
    #     feature_ext_model=senet,
    #     cfg=cfg.train
    # )
    loaded = torch.load(checkpoint_path)
    model.load_state_dict(loaded['state_dict'])
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

