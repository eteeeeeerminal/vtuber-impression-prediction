from typing import Optional, Sequence
from enum import Enum
import logging

from torch.utils.data import Dataset

from .label import DatasetPack, PathLike, VTuberInfos
from utils.logging import get_logger

class VTuberDataT(Enum):
    imp_all = "imp_all"
    yt_stat_all = "yt_stat_all"

yt_stat_types = set((
    VTuberDataT.yt_stat_all,
))

class VTuberDataset(Dataset):
    def __init__(self,
        json_path: PathLike, vtuber_data_t: VTuberDataT,
        vtuber_info_json_path: Optional[PathLike] = None,
        logger: Optional[logging.Logger] = None
        ) -> None:

        self.logger = get_logger(__name__) if logger is None else logger
        super().__init__()

        self.vtuber_data_t = vtuber_data_t
        self.vtuber_infos: Optional[VTuberInfos] = None
        if self.vtuber_data_t in yt_stat_types:
            assert isinstance(vtuber_info_json_path, PathLike)
            self.vtuber_infos = VTuberInfos(vtuber_info_json_path)

        self.logger.debug("Loading dataset labels.")
        self.dataset_label_data = DatasetPack.from_json(json_path)
        self.dataset_labels = self.dataset_label_data.dataset
        self.logger.debug("Done!")

    def get_label_dim(self) -> int:
        return len(self.get_label(0))

    def get_label(self, idx: int) -> Sequence[float]:
        data = self.dataset_labels[idx]
        if self.vtuber_data_t == VTuberDataT.imp_all:
            label = data.onom.impression + data.onom.personality + data.tipij.personality

        elif self.vtuber_data_t == VTuberDataT.yt_stat_all:
            label = self.vtuber_infos.get_yt_stat(data.origin.input.youtube_id)

        else:
            raise Exception("実装されていないデータタイプです。")

        return label

    def __getitem__(self, idx: int) -> list[float]:
        return self.get_label(idx)

    def __len__(self) -> int:
        return len(self.dataset_labels)
