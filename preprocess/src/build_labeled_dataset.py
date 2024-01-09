"""匿名化したデータに印象評価値を付与する"""

import os
import json
import pathlib
from dataclasses import asdict
import shutil

from dataset.onom_evaluation import OnomImpressionEvaluator
from dataset.data_types import (
    AnonymizedDataset, OnomVec, TIPIJVec, LabeledDataset, DatasetPack
)
from dataset.personality import OCEAN, set_personality_summarized

dataset_path = pathlib.Path("./data/annotated-dataset/input-and-label.json")
save_dir = pathlib.Path("./data/labeled-dataset")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

imp_db_path = pathlib.Path("./src/database/imp")
personality_db_path = pathlib.Path("./src/database/personality")
tipij_db_path = pathlib.Path("./src/database/TIPIJ")

# データセットの読み込み
with open(dataset_path, "r", encoding="utf-8") as f:
    json_dict = json.load(f)
    dataset = [AnonymizedDataset.from_dict(data) for data in json_dict]

# データベースの読み込み
onom_imp_evaluator = OnomImpressionEvaluator()
onom_imp_evaluator.load_db(imp_db_path)
onom_personality_evaluator = OnomImpressionEvaluator()
onom_personality_evaluator.load_db(personality_db_path)

# 匿名化したデータに印象評価値を付与する
def label_data(dataset: AnonymizedDataset) -> LabeledDataset:
    onom = dataset.label.first_onom
    onom_impression = onom_imp_evaluator.eval_onom_impression(onom)
    onom_personality = onom_personality_evaluator.eval_onom_impression(onom, False)
    onom_label = OnomVec(onom_impression, onom_personality, [])
    set_personality_summarized(onom_label, personality_db_path.joinpath("OCEAN.json"))

    # tipij の読み込み
    tipij_personality = [
        dataset.label.extroverted, dataset.label.critical,
        dataset.label.dependable, dataset.label.anxious,
        dataset.label.open, dataset.label.reserved,
        dataset.label.sympathetic, dataset.label.disorganized,
        dataset.label.calm, dataset.label.conventional
    ]
    tipij_personality = [(x - 4.0) / 3.0 for x in tipij_personality]
    tipij = TIPIJVec(tipij_personality, [])
    set_personality_summarized(tipij, tipij_db_path.joinpath("OCEAN.json"))

    # labeledDataset の保存
    return LabeledDataset(dataset.input.video_id, onom_label, tipij, dataset)

labeled_dataset = [label_data(data) for data in dataset]

# ラベルの読み込み
def load_scale_labels(path: pathlib.Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        scale = f.readlines()
        scale = list(map(lambda x: x.strip(), filter(lambda x: x, scale)))
        return scale

onom_imp_scale = load_scale_labels(imp_db_path.joinpath("sd_scale.csv"))
onom_personality_scale = load_scale_labels(personality_db_path.joinpath("sd_scale.csv"))
tipij_scale = load_scale_labels(tipij_db_path.joinpath("scale.csv"))

with open(save_dir.joinpath("labeled-dataset.json"), "w", encoding="utf-8") as f:
    save_data = DatasetPack(
        onom_imp_scale, onom_personality_scale, OCEAN, tipij_scale, OCEAN, labeled_dataset
    )
    json.dump(asdict(save_data), f, ensure_ascii=False, indent=4)
