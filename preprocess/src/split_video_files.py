# ダウンロードした大量の動画ファイルをアノテーション済みとアノテーションしてないやつに分ける

import argparse
import os
import re
import pathlib
import shutil
import json

parser = argparse.ArgumentParser()

parser.add_argument('-i')
parser.add_argument('-o')
parser.add_argument('-a')
args = parser.parse_args()
input_dir: str = args.i
output_dir: str = args.o
annotation_json_path: str = args.a

# アノテーションデータの読み込み
with open(annotation_json_path, "r", encoding="utf-8") as f:
    annotation_json = json.load(f)
    data_ids = set(map(lambda x: x["input"]["video_id"], annotation_json))

# input dir の検索と列挙
input_dir: pathlib.Path = pathlib.Path(input_dir)
video_files = list(input_dir.glob("*.mp4"))

# ファイル名からid抽出
def extract_id_from_file_name(path: str) -> str:
    match = re.match(r".+\[([\w-]+)\].mp4", path)
    return match[1]

# アノテーションあるやつとないやつに振り分け
def is_annotated(path: pathlib.Path) -> bool:
    video_id = extract_id_from_file_name(path.name)
    return video_id in data_ids

annotated = set()
not_annotated = set()
video_id_to_original_path: dict[str, pathlib.Path] = {}
for path in video_files:
    video_id = extract_id_from_file_name(path.name)
    video_id_to_original_path[video_id] = path

    # アノテーション済み
    if video_id in data_ids:
        annotated.add(video_id)

    # アノテーションしてない
    else:
        not_annotated.add(video_id)

# ファイルのコピー
output_dir: pathlib.Path = pathlib.Path(output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

annotated_dir = output_dir.joinpath("annotated")
not_annotated_dir = output_dir.joinpath("not_annotated")

if not os.path.exists(annotated_dir):
    os.mkdir(annotated_dir)

if not os.path.exists(not_annotated_dir):
    os.mkdir(not_annotated_dir)

already_exist_annotated = set(map(lambda x: x.name.replace(".mp4", ""), annotated_dir.glob("*.mp4")))
already_exist_not_annotated = set(map(lambda x: x.name.replace(".mp4", ""), not_annotated_dir.glob("*.mp4")))

## 不要なファイルの削除
should_delete_files= already_exist_annotated - annotated
for path in should_delete_files:
    os.remove(annotated_dir.joinpath(path + ".mp4"))

should_delete_files= already_exist_not_annotated - not_annotated
for path in should_delete_files:
    os.remove(not_annotated_dir.joinpath(path + ".mp4"))

## ファイルのコピー
print("copying video files. take a while")
should_copy_ids = annotated - already_exist_annotated
for video_id in should_copy_ids:
    original_path = video_id_to_original_path[video_id]
    shutil.copyfile(original_path, annotated_dir.joinpath(video_id + ".mp4"))

should_copy_ids = not_annotated - already_exist_not_annotated
for video_id in should_copy_ids:
    original_path = video_id_to_original_path[video_id]
    shutil.copyfile(original_path, not_annotated_dir.joinpath(video_id + ".mp4"))

# 動画ファイルが揃っているか確認
already_exist_annotated = set(map(lambda x: x.name, annotated_dir.glob("*.mp4")))
print("check annotated videos")
print(f"annotation data num: {len(data_ids)}")
print(f"found video num: {len(already_exist_annotated)}")
for data_id in data_ids:
    if data_id + ".mp4" not in already_exist_annotated:
        print(f"{data_id} is not found!")
print("done!")
