"""VTuberの自己紹介動画のフレーム画像を出すデータセット"""

import random
import pathlib
from dataclasses import dataclass
from typing import Sequence, Optional

import torch
import numpy as np
import albumentations
from PIL import Image

from .vtuber_dataset import VTuberDataset, VTuberDataT

@dataclass
class ImgDatasetConfig:
    imgs_dir: str
    img_ext: str
    label_path: str
    label_type: str
    max_frame: int
    vtuber_info_path: Optional[str] = None

IMG_SIZE = 256
INPUT_IMG_SIZE = 224

train_trans = albumentations.Compose([
    albumentations.Resize(IMG_SIZE, IMG_SIZE),
    albumentations.RandomCrop(INPUT_IMG_SIZE, INPUT_IMG_SIZE, p=0.7),
    albumentations.Resize(INPUT_IMG_SIZE, INPUT_IMG_SIZE),
    albumentations.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.5),
    albumentations.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.5),
])

test_trans = albumentations.Compose([
    albumentations.Resize(INPUT_IMG_SIZE, INPUT_IMG_SIZE),
])

def train_transforms(img) -> torch.Tensor:
    img = np.array(img, dtype=np.uint8)
    img = train_trans(image=img)['image']
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float32)
    img -= [128.0, 128.0, 128.0]
    img = img.transpose(2, 0, 1)  # C x H x W
    img = torch.from_numpy(img).float()
    return img

def test_transforms(img) -> torch.Tensor:
    img = np.array(img, dtype=np.uint8)
    img = test_trans(image=img)['image']
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float32)
    img -= [128.0, 128.0, 128.0]
    img = img.transpose(2, 0, 1)  # C x H x W
    img = torch.from_numpy(img).float()
    return img

class VTuberImgDataset(VTuberDataset):
    def __init__(self, cfg: ImgDatasetConfig, transforms) -> None:
        super().__init__(cfg.label_path, VTuberDataT(cfg.label_type), cfg.vtuber_info_path)
        self.cfg = cfg

        self.transforms = transforms

        self.logger.debug("ラベルに対応する画像データが存在するか確認します。")
        self.imgs_dir = pathlib.Path(cfg.imgs_dir)
        self.dataset_paging = []
        self.video_id_to_frame_list: list[list[int]] = []
        total_i = 0
        for i, data in enumerate(self.dataset_labels):
            # データなければログ出してスキップ
            img_dir = self.imgs_dir.joinpath(data.input_id)
            frame_num = len(list(img_dir.glob("*" + cfg.img_ext)))
            if frame_num <= 0:
                self.logger.warning(f"ID: {data.input_id} ラベルに対応する画像データが見つからないのでスキップします。")
                continue

            pages = [(i, k) for k in range(frame_num)]
            if len(pages) > self.cfg.max_frame:
                # 多すぎると困るのでランダムに選ぶ
                pages = random.sample(pages, self.cfg.max_frame)

            self.dataset_paging.extend(pages)
            frame_num = len(pages)

            frame_indices = list(range(total_i, total_i + frame_num))
            self.video_id_to_frame_list.append(frame_indices)
            total_i += frame_num

        self.logger.debug("Done!")
        self.logger.info("データの読み込みが終了しました。")

    def get_dataset_indices_by_video_indices(self, video_indices: Sequence[int]) -> list[int]:
        dataset_indices = [self.video_id_to_frame_list[i] for i in video_indices]
        dataset_indices = sum(dataset_indices, [])
        return dataset_indices

    def get_video_ids_len(self) -> int:
        return len(self.video_id_to_frame_list)

    def get_header_label() -> list[str]:
        raise Exception("未実装")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        video_i, frame_i = self.dataset_paging[idx]
        video_id = self.dataset_labels[video_i].input_id
        img_path = self.imgs_dir.joinpath(video_id).joinpath(str(frame_i) + self.cfg.img_ext)
        X = Image.open(img_path)
        if self.transforms is not None:
            X = self.transforms(X)

        label = super().__getitem__(video_i)
        label = torch.Tensor(label)
        return X, label

    def __len__(self) -> int:
        return len(self.dataset_paging)
