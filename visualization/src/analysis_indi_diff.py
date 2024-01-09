import os
import pathlib
import argparse
import json
import enum
from collections import Counter

from scipy.stats import pearsonr, spearmanr
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import japanize_matplotlib

from dataset import DatasetPack, LabeledDataset
from utils import cronback_alpha

sns.set_theme()
sns.set(font_scale=1.4)
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['IPAexGothic']

parser = argparse.ArgumentParser()
parser.add_argument("-i", default="../preprocess-for-ml/data/labeled-dataset/labeled-dataset.json", type=str)
parser.add_argument("-v", default="./data/merged.json", type=str) # VTuber データセット
parser.add_argument("-o", default="./output/indi_diff", type=str)
args = parser.parse_args()

dataset_path = pathlib.Path(args.i)
vtuber_data_path = pathlib.Path(args.v)
output_dir = pathlib.Path(args.o)

os.makedirs(output_dir, exist_ok=True)

dataset = DatasetPack.from_json(dataset_path)
with open(vtuber_data_path, "r", encoding="utf-8") as f:
    vtuber_data = json.load(f)

# 個人差の分析
## 被っているやつを列挙してみる
dataset_ids_count = Counter(map(lambda l: l.input_id, dataset.dataset))
data_ids = dataset_ids_count.keys()
print(len(dataset_ids_count))
dup_data_ids = set(filter(lambda i: dataset_ids_count[i] > 1, data_ids))
dup_dataset = list(filter(lambda l: l.input_id in dup_data_ids, dataset.dataset))
id_to_dup_dataset: dict[str, list[LabeledDataset]] = {}
for data in dup_dataset:
    if data.input_id in id_to_dup_dataset:
        id_to_dup_dataset[data.input_id][1] = data
    else:
        id_to_dup_dataset[data.input_id] = [data, None]

print(f"{len(dup_dataset)}")

right_df = pd.DataFrame(
    [
        data[0].onom.impression + data[0].onom.personality + data[0].tipij.personality
        for data in id_to_dup_dataset.values()
    ]
)
left_df = pd.DataFrame(
    [
        data[1].onom.impression + data[1].onom.personality + data[1].tipij.personality
        for data in id_to_dup_dataset.values()
    ]
)

mse = ((right_df - left_df)**2).mean(axis=1).mean()
((right_df - left_df)**2).to_csv("hoge.csv")
print(f"二乗誤差: {mse}")


for i in id_to_dup_dataset.keys():
    name = id_to_dup_dataset[i][0].origin.input.youtube_name
    f_onom = [data.origin.label.first_onom for data in id_to_dup_dataset[i]]
    o_onom = [data.origin.label.other_onom for data in id_to_dup_dataset[i]]
    o_imp = [data.origin.label.other_impressions for data in id_to_dup_dataset[i]]
    print(f"{name} & オノマトペ（1語で） & {f_onom[0]} & {f_onom[1]} \\\\")
    print(f" & オノマトペ（上記以外） & {o_onom[0]} & {o_onom[1]} \\\\")
    print(f" & オノマトペ以外の印象 & {o_imp[0]} & {o_imp[1]} \\\\")
    print("\hline")

# オノマトペ質感の相関を調べる
def p_to_star(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""

fig, axes = plt.subplots(9, 5, figsize=(18,24))
for i in range(43):
    a = []
    b = []
    for data in id_to_dup_dataset.values():
        a.append(data[0].onom.impression[i])
        b.append(data[1].onom.impression[i])

    ax = axes[int(i/5)][i%5]
    ax.set_title(dataset.onom_imp_scale[i])
    ax.axline((-0.8, -0.8), (0.8, 0.8), c=".2", ls="--", zorder=0)
    sns.scatterplot(x=a, y=b, ax=ax)
    smr = spearmanr(a, b)
    label = f"{dataset.onom_imp_scale[i]}"
    r = f"{smr.statistic:1.3f}"
    p = f"{smr.pvalue:1.3f}{p_to_star(smr.pvalue)}"
    print(f"{label} & {r} & {p} \\\\")

fig.tight_layout()
plt.savefig(output_dir.joinpath("onom_imp.png"))
plt.close()

print("----")

fig, axes = plt.subplots(10, 5, figsize=(18,24))
for i in range(50):
    a = []
    b = []
    for data in id_to_dup_dataset.values():
        a.append(data[0].onom.personality[i])
        b.append(data[1].onom.personality[i])

    ax = axes[int(i/5)][i%5]
    ax.set_title(dataset.onom_personality_scale[i])
    ax.axline((-0.8, -0.8), (0.8, 0.8), c=".2", ls="--", zorder=0)
    sns.scatterplot(x=a, y=b, ax=ax)
    smr = spearmanr(a, b)
    label = f"{dataset.onom_personality_scale[i]}"
    r = f"{smr.statistic:1.3f}"
    p = f"{smr.pvalue:1.3f}{p_to_star(smr.pvalue)}"
    print(f"{label} & {r} & {p} \\\\")

fig.tight_layout()
plt.savefig(output_dir.joinpath("onom_per.png"))
plt.close()

print("----")

fig, axes = plt.subplots(3, 4, figsize=(18,9))
for i in range(10):
    a = []
    b = []
    for data in id_to_dup_dataset.values():
        a.append(data[0].tipij.personality[i])
        b.append(data[1].tipij.personality[i])

    ax = axes[int(i/4)][i%4]
    ax.set_title(dataset.tipij_scale[i])
    ax.axline((-0.8, -0.8), (0.8, 0.8), c=".2", ls="--", zorder=0)
    sns.scatterplot(x=a, y=b, ax=ax)
    smr = spearmanr(a, b)
    label = f"{dataset.tipij_scale[i]}"
    r = f"{smr.statistic:1.3f}"
    p = f"{smr.pvalue:1.3f}{p_to_star(smr.pvalue)}"
    print(f"{label} & {r} & {p} \\\\")

fig.tight_layout()
plt.savefig(output_dir.joinpath("tipij_per.png"))
plt.close()
