import os
import json
from dataclasses import dataclass

PathLike = str | bytes | os.PathLike

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

@dataclass
class AnnotatorAttributes:
    annotator_id: str
    watch_frequency: str
    many_v_watch: str
    watch_period: str
    sex: str
    age: str
    platform_check: list[str]
    sns_check: list[str]
    annotation_num: int

    @classmethod
    def from_json(cls, json_path: PathLike):
        with open(json_path, "r", encoding="utf-8") as f:
            data_json = json.load(f)
            return [cls(**data) for data in data_json]

