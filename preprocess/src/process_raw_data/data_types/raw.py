import json
from dataclasses import dataclass
import datetime

from ..utils import PathLike

@dataclass
class Metadata:
    user_id: str
    display_name: str
    data_id: str
    timestamp: int

    @classmethod
    def from_json(cls, json_dict: dict):
        return cls(
            user_id = json_dict["userId"],
            display_name = json_dict["displayName"],
            data_id = json_dict["dataId"],
            timestamp = json_dict["timestamp"]
        )

@dataclass
class ContentCommon:
    dataset_version: str
    consent_check: list[str]
    consent_radio: str
    name: str
    email: str
    watch_frequency: str
    many_v_watch: str
    watch_period: str
    sex: str
    age: str
    platform_check: list[str]
    sns_check: list[str]

    @classmethod
    def from_json(cls, json_dict: dict):
        return cls(
            dataset_version = json_dict["datasetVersion"],
            consent_check = json_dict["consentCheck"],
            consent_radio = json_dict["consentRadio"],
            name = json_dict["name"],
            email = json_dict["email"],
            watch_frequency = json_dict["watchFrequency"],
            many_v_watch = json_dict["manyVWatch"],
            watch_period = json_dict["watchPeriod"],
            sex = json_dict["sex"],
            age = json_dict["age"],
            platform_check = json_dict["platformCheck"],
            sns_check = json_dict["snsCheck"]
        )

@dataclass
class ContentVtuber:
    dataset_version: str
    already_know: str
    first_onom: str
    other_onom: str
    other_impressions: str
    extroverted: str
    critical: str
    dependable: str
    anxious: str
    open: str
    reserved: str
    sympathetic: str
    disorganized: str
    calm: str
    conventional: str

    @classmethod
    def from_json(cls, json_dict: dict):
        return cls(
            dataset_version = json_dict["datasetVersion"],
            already_know = json_dict["alreadyKnow"],
            first_onom = json_dict["firstOnomatopoeia"],
            other_onom = json_dict["otherOnomatopoeia"],
            other_impressions = json_dict["otherImpressions"],
            extroverted = json_dict["extroverted"],
            critical = json_dict["critical"],
            dependable = json_dict["dependable"],
            anxious = json_dict["anxious"],
            open = json_dict["open"],
            reserved = json_dict["reserved"],
            sympathetic = json_dict["sympathetic"],
            disorganized = json_dict["disorganized"],
            calm = json_dict["calm"],
            conventional = json_dict["conventional"]
        )

@dataclass
class FormValue:
    metadata: Metadata
    content: ContentCommon | ContentVtuber

    @classmethod
    def from_json(cls, json_dict: dict):
        content: dict = json_dict["content"]
        if content.get("consentRadio"):
            content = ContentCommon.from_json(content)
        else:
            content = ContentVtuber.from_json(content)

        return cls(
            metadata = Metadata.from_json(json_dict["metadata"]),
            content = content
        )

def load_form_values(path: PathLike) -> list[FormValue]:
    with open(path, "r", encoding="utf-8") as f:
        form_values = json.load(f)
        return [FormValue.from_json(v) for v in form_values]
