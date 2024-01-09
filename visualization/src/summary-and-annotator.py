import re
import os
import pathlib
import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import seaborn.objects as so
import japanize_matplotlib
from jaconv.jaconv import kata2hira

from dataset import DatasetPack, AnnotatorAttributes

sns.set_theme()
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['IPAexGothic']

parser = argparse.ArgumentParser()
parser.add_argument("-l", default="../preprocess-for-ml/data/labeled-dataset/labeled-dataset.json", type=str)
parser.add_argument("-a", default="../preprocess-for-ml/data/annotated-dataset/annotator.json", type=str)
parser.add_argument("-v", default="./data/merged.json", type=str) # VTuber データセット
parser.add_argument("-o", default="./output/annotator", type=str)
args = parser.parse_args()

dataset_path = pathlib.Path(args.l)
annotator_path = pathlib.Path(args.a)
vtuber_data_path = pathlib.Path(args.v)
output_dir = pathlib.Path(args.o)

os.makedirs(output_dir, exist_ok=True)

dataset = DatasetPack.from_json(dataset_path)
with open(vtuber_data_path, "r", encoding="utf-8") as f:
    vtuber_data = json.load(f)
annotator_data = AnnotatorAttributes.from_json(annotator_path)

print(f"データ数: {len(dataset.dataset)}")
print(f"被験者の数: {len(annotator_data)}")
annotator_ages = [d.age for d in annotator_data]
annotator_ages = Counter(annotator_ages)
print(f"被験者の年代: {annotator_ages}")
annotator_sex = [d.sex for d in annotator_data]
annotator_sex = Counter(annotator_sex)
print(f"被験者の性別: {annotator_sex}")
first_onoms = list(map(lambda x: kata2hira(x.origin.label.first_onom), dataset.dataset))
first_onoms_counter = Counter(first_onoms)
other_onoms = map(lambda x: re.split("[、 　]", x.origin.label.other_onom), dataset.dataset)
other_onoms = sum(other_onoms, [])
other_onoms = list(filter(lambda x: x, map(kata2hira, other_onoms)))
all_onoms_counter = Counter(first_onoms+other_onoms)
print(f"オノマトペの例: {list(first_onoms_counter.most_common())[:5]}")
print(f"第1オノマトペの種類: {len(first_onoms_counter.keys())}")
print(f"全部のオノマトペの種類: {len(all_onoms_counter.keys())}")

# 被験者の基本統計を見てみる
def anno_plot(df, filename, y):
    fig, ax = plt.subplots()
    plot = so.Plot(df, x="回答数", y=y).add(so.Bar()).on(ax)
    plot.plot(True)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    plt.savefig(output_dir.joinpath(filename))
    plt.close()

## 性別
sex_h = ["男性", "女性"]
anno_plot(pd.DataFrame([[annotator_sex.get(h, 0), h] for h in sex_h], columns=["回答数", "性別"]), "sex.png", "性別")

## 年齢
age_h = ["10代", "20代"]
anno_plot(pd.DataFrame([[annotator_ages.get(h, 0), h] for h in age_h], columns=["回答数", "年齢"]), "age.png", "年齢")


## VTuber の視聴態度的なやつ
pd_header = ["視聴頻度", "視聴歴", "視聴人数"]
def annotator_to_list(attr: AnnotatorAttributes) -> list:
    return [attr.watch_frequency, attr.watch_period, attr.many_v_watch]
anno_df = pd.DataFrame(
    map(annotator_to_list, annotator_data), columns=pd_header
)

def anno_v_plot(attr: str, header: list[str], filename: str):
    df = anno_df[attr]
    counter = Counter(df)
    df = pd.DataFrame(
        [[counter.get(h, 0), h] for h in header], columns=["回答数", "選択肢"]
    )
    fig, ax = plt.subplots()
    plot = so.Plot(df, x="回答数", y="選択肢").add(so.Bar()).on(ax)
    plot.plot(True)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    plt.savefig(output_dir.joinpath(filename))
    plt.close()

## 視聴頻度
anno_v_plot("視聴頻度", ["2週間1回以下", "週1回", "週2,3回", "2日に1回", "毎日"], "freq.png")

## 視聴人数
anno_v_plot("視聴人数", ["いない", "1~4人程度", "5~10人程度", "10~30人程度", "30~50人程度", "50人~"], "many.png")

## 視聴歴
anno_v_plot("視聴歴", ["~1ヶ月", "1ヶ月~半年", "半年~1年", "1年~1年半", "1年半~2年", "2年~"], "period.png")

## 複数回答系はカウント方法が違うので別途
## 動画プラットフォーム
vplat_list = filter(lambda x: x, sum(map(lambda x: x.platform_check, annotator_data), []))
vplat_counter = Counter(vplat_list)
vplat_h = [
    "YouTube", "NicoNico動画", "Twitch", "BiliBili動画",
    "TikTok", "その他", "見ない"
]
print(vplat_counter)
vplat_counter["その他"] = vplat_counter["その他の動画プラットフォーム"]
vplat_df = pd.DataFrame(
    [[vplat_counter.get(h, 0), h] for h in vplat_h], columns=["回答数", "選択肢"]
)

fig, ax = plt.subplots()
plot = so.Plot(vplat_df, x="回答数", y="選択肢").add(so.Bar()).on(ax)
plot.plot(True)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
fig.tight_layout()
plt.savefig(output_dir.joinpath("annotator_vplat.png"))
plt.close()

## SNS
sns_list = filter(lambda x: x, sum(map(lambda x: x.sns_check, annotator_data), []))
sns_counter = Counter(sns_list)
sns_h = [
    "Twitter", "Facebook", "Instagram", "VRChat 等の VRSNS",
    "その他", "使わない"
]
print(sns_counter)
sns_counter["その他"] = sns_counter["その他のSNS"]
vplat_df = pd.DataFrame(
    [[sns_counter.get(h, 0), h] for h in sns_h], columns=["回答数", "選択肢"]
)

fig, ax = plt.subplots()
plot = so.Plot(vplat_df, x="回答数", y="選択肢").add(so.Bar()).on(ax)
plot.plot(True)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
fig.tight_layout()
plt.savefig(output_dir.joinpath("annotator_sns.png"))
plt.close()

