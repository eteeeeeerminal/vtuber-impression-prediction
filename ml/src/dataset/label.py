import json
import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from utils import PathLike
from utils.logging import get_logger

@dataclass
class AnnotationInput:
    youtube_id: str
    youtube_name: str
    video_id: str
    video_name: str

@dataclass
class AnnotationLabel:
    annotator_id: str
    already_know: str
    first_onom: str
    other_onom: str
    other_impressions: str
    extroverted: int
    critical: int
    dependable: int
    anxious: int
    open: int
    reserved: int
    sympathetic: int
    disorganized: int
    calm: int
    conventional: int

@dataclass
class AnonymizedDataset:
    input: AnnotationInput
    label: AnnotationLabel

    @classmethod
    def from_dict(cls, json_dict: dict):
        input = AnnotationInput(**json_dict["input"])
        label = AnnotationLabel(**json_dict["label"])
        return cls(input, label)

@dataclass
class OnomVec:
    # それぞれ -1~1
    impression: list[float]
    personality: list[float]
    personality_summarized: list[float]

@dataclass
class TIPIJVec:
    # それぞれ -1~1
    personality: list[float]
    personality_summarized: list[float]

@dataclass
class LabeledDataset:
    input_id: str
    onom: OnomVec
    tipij: TIPIJVec
    origin: AnonymizedDataset

    @classmethod
    def from_dict(cls, json_dict: dict):
        return cls(
            json_dict["input_id"],
            OnomVec(**json_dict["onom"]),
            TIPIJVec(**json_dict["tipij"]),
            AnonymizedDataset.from_dict(json_dict["origin"])
        )

# dataset label
@dataclass
class DatasetPack:
    onom_imp_scale: list[str]
    onom_personality_scale: list[str]
    onom_personality_summarized_scale: list[str]
    tipij_scale: list[str]
    tipij_summarized_scale: list[str]
    dataset: list[LabeledDataset]

    @classmethod
    def from_dict(cls, json_dict: dict):
        return cls(
            json_dict["onom_imp_scale"], json_dict["onom_personality_scale"],
            json_dict["onom_personality_summarized_scale"],
            json_dict["tipij_scale"], json_dict["tipij_summarized_scale"],
            [LabeledDataset.from_dict(data) for data in json_dict["dataset"]]
        )

    @classmethod
    def from_json(cls, json_path: PathLike):
        with open(json_path, "r", encoding="utf-8") as f:
            data_json = json.load(f)
            return cls.from_dict(data_json)

    def normalize(self):
        # 被っているデータを削除する
        found_id: set[str] = set()
        normalized_dataset: list[LabeledDataset] = []

        for data in self.dataset:
            if data.input_id in found_id:
                continue # すでにアノテーションのラベルが見つかっているデータ
            else:
                normalized_dataset.append(data)
                found_id.add(data.input_id)

        self.dataset = normalized_dataset

class VTuberInfos:
    def __init__(self, json_path: PathLike, logger: Optional[logging.Logger] = None) -> None:
        self.logger = get_logger(__name__) if logger is None else logger

        with open(json_path, "r", encoding="utf-8") as f:
            data_json = json.load(f)
            self.vtuber_infos = {data["vtuber_id"]: data for data in data_json}

        self.id_to_index = {}
        self.vtuber_stat = []
        i = 0
        for v_id, data in self.vtuber_infos.items():
            info = data["youtube"]
            yt_stat = (info["subscriber_count"], info["view_count"])
            if not (isinstance(yt_stat[0], int) and yt_stat[0] > 0 \
                and isinstance(yt_stat[1], int) and yt_stat[1] > 0):
                continue

            self.id_to_index[v_id] = i
            self.vtuber_stat.append(yt_stat)
            i += 1

        self.vtuber_stat = np.array(self.vtuber_stat)
        self.vtuber_stat = np.log10(self.vtuber_stat)
        self.vtuber_stat = (self.vtuber_stat - self.vtuber_stat.mean()) / self.vtuber_stat.std()

    def get_yt_stat(self, vtuber_id: str) -> Sequence[float]:
        return self.vtuber_stat[self.id_to_index[vtuber_id]]
