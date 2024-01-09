"""VTuber の自己紹介動画の音声データを読み込むデータセット"""

import os
import random
import pathlib
from dataclasses import dataclass
from typing import Sequence, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .label import DatasetPack
from .vtuber_dataset import VTuberDataset, VTuberDataT

@dataclass
class AudioDatasetConfig:
    audios_dir: str
    label_path: str
    label_type: str
    label_transform: Optional[str]
    vec_num: int # 1つのデータにまとめるベクトルの数
    min_vec: int #
    max_vec: int # 1つの動画からとるベクトルの最大数
    vtuber_info_path: Optional[str] = None

class VTuberAudioDataset(VTuberDataset):
    def __init__(self, cfg: AudioDatasetConfig) -> None:
        super().__init__(cfg.label_path, VTuberDataT(cfg.label_type), cfg.vtuber_info_path)
        self.cfg = cfg

        self.logger.debug("ラベルに対応する音声データを読み込んでいます。")
        self.audios_dir = pathlib.Path(cfg.audios_dir)
        self.dataset_paging = []
        self.video_id_to_feature_list = []
        self.video_id_to_feature_data = []
        total_i = 0
        video_i = 0
        for i, data in enumerate(self.dataset_labels):
            data_path = self.audios_dir.joinpath(data.input_id+".txt")
            if not os.path.exists(data_path):
                self.logger.warning(f"ID: {data.input_id} ラベルに対応する音声データが見つからないのでスキップします。")
                continue

            input_data: np.ndarray = np.loadtxt(data_path)
            found_vec_n = input_data.shape[0] - self.cfg.vec_num + 1
            vec_indices = list(range(found_vec_n))

            if found_vec_n < self.cfg.min_vec:
                self.logger.warning(f"ID: {data.input_id} 音声データの長さが不十分なのでスキップします。")
                continue

            elif found_vec_n > self.cfg.max_vec:
                vec_indices = random.sample(vec_indices, self.cfg.max_vec)

            self.dataset_paging.extend([(video_i, k) for k in vec_indices])
            vec_num = len(vec_indices)

            vec_indices = list(range(total_i, total_i + vec_num))
            self.video_id_to_feature_list.append(vec_indices)
            self.video_id_to_feature_data.append(input_data)
            total_i += vec_num
            video_i += 1

        self.logger.debug("Done!")
        self.logger.info("データの読み込みが終了しました。")

    def get_dataset_indices_by_video_indices(self, video_indices: Sequence[int]) -> list[int]:
        dataset_indices = [self.video_id_to_feature_list[i] for i in video_indices]
        dataset_indices = sum(dataset_indices, [])
        return dataset_indices

    def get_feature_dim(self) -> int:
        x, _ = self.__getitem__(0)
        return x.shape[0]

    def get_video_ids_len(self) -> int:
        return len(self.video_id_to_feature_data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        video_i, vec_i = self.dataset_paging[idx]
        input_data = self.video_id_to_feature_data[video_i]

        X = torch.Tensor(input_data[vec_i: vec_i+self.cfg.vec_num]).flatten()

        label = super().__getitem__(video_i)
        label = torch.Tensor(label)

        if self.cfg.label_transform is None:
            pass

        elif self.cfg.label_transform == "prob":
            label = (label + 1.0) / 2.0
            label = torch.where(label > 1.0, 1.0, label)
            label = torch.where(label < 0.0, 0.0, label)

        return X, label

    def __len__(self) -> int:
        return len(self.dataset_paging)
