# 実験の手続きに必要な書類等を生成するスクリプト群

import datetime
from .data_types.shaped import ShapedData, Annotation
from .utils import PathLike
from .email_templete import (
    email_title, email_intro_templete,
    email_ok_templete, email_ng_templete,
    email_end_templete
)

# 同意内容の取得
min_annotation_num = 6

# 参加者がどれくらいアノテーションしたか, 謝金支払い額を出す。
## uid, 名前, 性別, email, 合計アノテーション数, 合計実験時間見積もり, 謝金額, 各実施日にいくつアノテーションしたか
def _schedule(annotations: list[Annotation]) -> str:
    annotations = sorted(annotations, key = lambda x: x.metadata.timestamp)
    counts = {}
    for anno in annotations:
        date = datetime.datetime.fromtimestamp(anno.metadata.timestamp / 1000)
        date = date.strftime("%m/%d")
        if date in counts:
            counts[date] += 1
        else:
            counts[date] = 1

    return ", ".join([f"{k}: {v}" for k, v in counts.items()])


def shaped_data_to_reward_data_row(shaped_data: ShapedData) -> list[str]:
    uid = shaped_data.uid
    name = shaped_data.consent.name
    sex = shaped_data.user_attr.sex
    email = shaped_data.consent.email
    anno_num = len(shaped_data.annotations)
    anno_time = anno_num / 6.0

    if anno_num < min_annotation_num:
        reward = "0 (規定数に満たず)"
    else:
        reward = int(anno_num / 3) * 500

    schedule = _schedule(shaped_data.annotations)

    ret_row = [uid, name, sex, email, anno_num, anno_time, reward, schedule]
    return [str(e) for e in ret_row]

def output_reward_data(shaped_datum: dict[str, ShapedData], output_tsv: PathLike):
    rows: list[list[str]] = [
        shaped_data_to_reward_data_row(data) for data in shaped_datum.values()
    ]

    # save
    header = ["uid", "名前", "性別", "email", "合計アノテーション数", "合計実験時間(見積もり)", "謝金額", "実施状況"]
    header = "\t".join(header)
    with open(output_tsv, "w", encoding="utf-8") as f:
        rows = ["\t".join(r) for r in rows]
        f.write(header + "\n")
        f.write("\n".join(rows))

# メール生成
## アノテーション数6以下の人にはお断りのメール
## アノテーション数0の人は無視
## アノテーション数6以上の人には謝金まわりうんぬんのメール
def shaped_data_to_email_txt(shaped_data: ShapedData) -> str:
    anno_num = len(shaped_data.annotations)
    if anno_num == 0:
        return ""

    email_txt = "---\n"
    email_txt += f"{shaped_data.consent.email}\n"
    email_txt += "\n"
    email_txt += email_intro_templete.replace("{name}", shaped_data.consent.name)
    email_txt += "\n"

    reward = int(anno_num / 3) * 500
    if anno_num < min_annotation_num:
        email_body_txt = email_ng_templete
    else:
        email_body_txt = email_ok_templete

    email_body_txt = email_body_txt.replace("{name}", shaped_data.consent.name)
    email_body_txt = email_body_txt.replace("{anno_count}", str(anno_num))
    email_body_txt = email_body_txt.replace("{reward}", str(reward))
    email_txt += email_body_txt
    email_txt += "\n"
    email_txt += email_end_templete
    email_txt += "\n"

    email_txt += "---\n"

    return email_txt

def output_email_txt(shaped_datum: dict[str, ShapedData], output_txt: PathLike):
    email_txt = email_title + "\n"
    email_txt += "\n".join([shaped_data_to_email_txt(v) for v in shaped_datum.values()])

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(email_txt)
