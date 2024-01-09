# 被験者の回答ログから同意ログを作成する
## 各被験者 1json 的な感じでやりたい
## 提出用の1行1jsonのものも作成
## 名前, email, 回答日時, 同意6項目, 同意したかどうか
import argparse
import json
from dataclasses import asdict
from process_raw_data.data_types.raw import load_form_values, ContentCommon
from process_raw_data.data_types.shaped import Consent

parser = argparse.ArgumentParser()

parser.add_argument('-i')
parser.add_argument('-f', default="./data/include_ids.json")
parser.add_argument('-o')
args = parser.parse_args()
input_json: str = args.i
output_json: str = args.o
include_ids_json: str = args.f

with open(include_ids_json, "r", encoding="utf-8") as f:
    checked_list = set(json.load(f))

form_values = load_form_values(input_json)
common_form_values = filter(lambda x: type(x.content) == ContentCommon, form_values)
checked_values = filter(lambda x: x.metadata.user_id in checked_list, common_form_values)
consent_values = [Consent.from_form_value(v) for v in checked_values]

with open(output_json, "w", encoding="utf-8") as f:
    save_obj = [asdict(v) for v in consent_values]
    json.dump(save_obj, f, ensure_ascii=False, indent=4)
