import random
import json
from dataclasses import dataclass

from .raw import FormValue, Metadata, ContentCommon, ContentVtuber
from ..utils import PathLike, timestamp_to_str

@dataclass
class Consent:
    name: str
    email: str
    timestamp: str
    consent_check: list[str]
    consent_radio: str

    @classmethod
    def from_form_value(cls, form_value: FormValue):
        assert type(form_value.content) == ContentCommon
        return cls(
            name = form_value.content.name,
            email = form_value.content.email,
            timestamp = timestamp_to_str(form_value.metadata.timestamp),
            consent_check = form_value.content.consent_check,
            consent_radio = form_value.content.consent_radio
        )

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
    def from_form_value(cls, form_value: FormValue):
        content = form_value.content
        assert type(content) == ContentCommon
        return cls(
            annotator_id = form_value.metadata.user_id,
            watch_frequency = content.watch_frequency,
            many_v_watch = content.many_v_watch,
            watch_period = content.watch_period,
            sex = content.sex,
            age = content.age,
            platform_check = content.platform_check,
            sns_check = content.sns_check,
            annotation_num = 0
        )

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

    @classmethod
    def from_form_value(cls, form_value: FormValue):
        content = form_value.content
        assert type(content) == ContentVtuber
        return cls(
            annotator_id = form_value.metadata.user_id,
            already_know = content.already_know,
            first_onom = content.first_onom,
            other_onom = content.other_onom,
            other_impressions = content.other_impressions,
            extroverted = int(content.extroverted),
            critical = int(content.critical),
            dependable = int(content.dependable),
            anxious = int(content.anxious),
            open = int(content.open),
            reserved = int(content.reserved),
            sympathetic = int(content.sympathetic),
            disorganized = int(content.disorganized),
            calm = int(content.calm),
            conventional = int(content.conventional)
        )

@dataclass
class AnnotationInput:
    youtube_id: str
    youtube_name: str
    video_id: str
    video_name: str

    @classmethod
    def from_dataset_json(cls, dataset_json: dict):
        youtube = dataset_json["youtube"]
        video = youtube["target_video"]
        return cls(
            youtube_id = youtube["channel_id"],
            youtube_name = youtube["name"],
            video_id = video["video_id"],
            video_name = video["title"]
        )

@dataclass
class Annotation:
    metadata: Metadata
    label: AnnotationLabel
    input: AnnotationInput | None = None

    @classmethod
    def from_form_value(cls, form_value: FormValue):
        return cls(
            metadata = form_value.metadata,
            label = AnnotationLabel.from_form_value(form_value),
        )

@dataclass
class ShapedData:
    uid: str
    consent: Consent
    user_attr: AnnotatorAttributes
    annotations: list[Annotation]

    @classmethod
    def from_form_value(cls, form_value: FormValue):
        assert type(form_value.content) == ContentCommon
        return cls(
            uid = form_value.metadata.user_id,
            consent = Consent.from_form_value(form_value),
            user_attr = AnnotatorAttributes.from_form_value(form_value),
            annotations = []
        )

    def add_annotation(self, form_value: FormValue):
        annotation = Annotation.from_form_value(form_value)
        self.annotations.append(annotation)

def load_shaped_datum(path: PathLike) -> dict[str, ShapedData]:
    with open(path, "r", encoding="utf-8") as f:
        form_values = json.load(f)
        form_values = [FormValue.from_json(v) for v in form_values]
        shaped_datum: dict[str, ShapedData] = {}

        for value in form_values:
            if type(value.content) == ContentCommon:
                shaped = ShapedData.from_form_value(value)
                assert shaped.uid not in shaped_datum
                shaped_datum[shaped.uid] = shaped

        for value in form_values:
            if type(value.content) == ContentVtuber:
                shaped_datum[value.metadata.user_id].add_annotation(value)

        for value in shaped_datum.values():
            value.user_attr.annotation_num = len(value.annotations)

    return shaped_datum

def anonymize_shaped_datum(
    shaped_datum: dict[str, ShapedData],
    author_id: str
) -> dict[str, ShapedData]:
    """破壊的メソッド, シャッフルして, uid を上書きで振りなおす"""
    keys = list(shaped_datum.keys())
    keys.remove(author_id)
    random.shuffle(keys)
    keys = [author_id] + keys
    anonymized_id_dict = {key: i for i, key in enumerate(keys)}

    ret_datum = {}
    for key in anonymized_id_dict.keys():
        value = shaped_datum[key]
        value.uid = anonymized_id_dict[value.uid]
        value.user_attr.annotator_id = anonymized_id_dict[value.user_attr.annotator_id]
        for anno in value.annotations:
            anno.label.annotator_id = anonymized_id_dict[anno.label.annotator_id]

        ret_datum[key] = value

    return ret_datum
