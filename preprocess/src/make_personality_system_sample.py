"""パーソナリティシステムの検証用のデータを作成する"""

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

origin_onom_path = pathlib.Path("./onom_list/origin.txt")
with open(origin_onom_path, 'r', encoding='utf-8') as f:
    origin_onoms = f.readlines()

komatsu_onom_path = pathlib.Path("./onom_list/komatsu.txt")
with open(komatsu_onom_path, 'r', encoding='utf-8') as f:
    komatsu_onoms = f.readlines()

save_path = pathlib.Path("./data/onom_personality_sample.json")

imp_db_path = pathlib.Path("./src/database/imp")
personality_db_path = pathlib.Path("./src/database/personality")
onom_imp_evaluator = OnomImpressionEvaluator()
onom_imp_evaluator.load_db(imp_db_path)
onom_personality_evaluator = OnomImpressionEvaluator()
onom_personality_evaluator.load_db(personality_db_path)

def onom_to_vec(onom: str) -> dict:
    imp = onom_imp_evaluator.eval_onom_impression(onom)
    per = onom_personality_evaluator.eval_onom_impression(onom, False)
    vec = OnomVec(imp, per, [])
    set_personality_summarized(vec, personality_db_path.joinpath("OCEAN.json"))
    return asdict(vec)

output_data = {}
output_data["origin"] = [onom_to_vec(onom) for onom in origin_onoms]
output_data["komatsu"] = [onom_to_vec(onom) for onom in komatsu_onoms]

with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)
