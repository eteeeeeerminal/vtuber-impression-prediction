# 匿名化したデータを作成

import os
import pathlib
import json
from dataclasses import asdict
from process_raw_data.data_types.raw import ContentCommon, load_form_values
from process_raw_data.data_types.shaped import (
    anonymize_shaped_datum, load_shaped_datum, ShapedData,
    AnnotationInput
)

min_annotation_num = 6
save_path = "./data/annotated-dataset"
dataset_path = "./data/dataset/vtuber-dataset.json"
ids_path = "./data/include_ids.json"
author_id = "r3Kbwjw0V2XkH83eX3zsONbSOSg1"

save_path = pathlib.Path(save_path)
if not os.path.exists(save_path):
    os.mkdir(save_path)

shaped_datum = load_shaped_datum("./data/12-19/vtuber-onomatopoeia.json")
with open(ids_path, "r", encoding="utf-8") as f:
    checked_list = set(json.load(f))

# 規定のアノテーション数以下の人は削除
filtered_shaped_datum: dict[str, ShapedData] = {}
for key in shaped_datum.keys():
    if len(shaped_datum[key].annotations) < min_annotation_num:
        # 少なすぎスキップ
        continue

    if key not in checked_list:
        # 謝金を受領してもらっていない
        continue

    filtered_shaped_datum[key] = shaped_datum[key]

# 匿名化
anonymized_datum = anonymize_shaped_datum(filtered_shaped_datum, author_id)

# アノテーション対象のデータ, input の読み込み
with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)
    dataset_dict = {data["vtuber_id"]: data for data in dataset}

for value in anonymized_datum.values():
    for anno in value.annotations:
        anno.input = AnnotationInput.from_dataset_json(
            dataset_dict[anno.metadata.data_id]
        )


# 保存
annotation_path = save_path.joinpath("input-and-label.json")
annotator_path = save_path.joinpath("annotator.json")

with open(annotation_path, "w", encoding="utf-8") as f:
    annotations = sum([
        value.annotations for value in anonymized_datum.values()
    ], [])
    annotations = [{
        "input": asdict(a.input), "label": asdict(a.label)
    } for a in annotations]
    print(len(annotations))
    json.dump(annotations, f, ensure_ascii=False, indent=4)

with open(annotator_path, "w", encoding="utf-8") as f:
    annotators = [
        asdict(value.user_attr) for value in anonymized_datum.values()
    ]
    json.dump(annotators, f, ensure_ascii=False, indent=4)
