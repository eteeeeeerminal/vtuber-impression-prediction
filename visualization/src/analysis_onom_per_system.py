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
from word_cloud import WordCloud

sns.set_theme()
sns.set(font_scale=1.4)
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['IPAexGothic']

parser = argparse.ArgumentParser()
parser.add_argument("-i", default="../preprocess-for-ml/data/onom_personality_sample.json", type=str)
parser.add_argument("-o", default="./output/onom_per_system", type=str)
args = parser.parse_args()

data_path = pathlib.Path(args.i)
output_dir = pathlib.Path(args.o)

os.makedirs(output_dir, exist_ok=True)

with open(data_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

origin_per_df = pd.DataFrame(
    [data["personality"] for data in dataset["origin"]]
)

komatsu_per_df = pd.DataFrame(
    [data["personality"] for data in dataset["komatsu"]]
)

all_per_df = pd.concat([origin_per_df, komatsu_per_df])

ocean_columns = [
    "O (開放性)", "C (誠実性)", "E (外向性)", "A (調和性)", "N (情緒不安定性)"
]
origin_ocean_df = pd.DataFrame(
    [data["personality_summarized"] for data in dataset["origin"]],
    columns=ocean_columns
)
komatsu_ocean_df = pd.DataFrame(
    [data["personality_summarized"] for data in dataset["komatsu"]],
    columns=ocean_columns
)

all_ocean_df = pd.concat([origin_ocean_df, komatsu_ocean_df])

"""
onom_per_corr = all_per_df.corr(method="spearman")
fig, ax = plt.subplots(figsize=(50,50))
sns.heatmap(onom_per_corr, xticklabels=False, yticklabels=False)
fig.tight_layout()
plt.savefig(output_dir.joinpath("all_per_corr.png"))
plt.close()

print(f"E: {cronback_alpha(all_per_df.iloc[:,  0:10])}")
print(f"A: {cronback_alpha(all_per_df.iloc[:, 10:20])}")
print(f"C: {cronback_alpha(all_per_df.iloc[:, 20:30])}")
print(f"N: {cronback_alpha(all_per_df.iloc[:, 30:40])}")
print(f"O: {cronback_alpha(all_per_df.iloc[:, 40:50])}")
"""

sns.pairplot(origin_ocean_df)
plt.savefig(output_dir.joinpath("origin_ocean_pairplot.png"))
plt.close()
sns.pairplot(komatsu_ocean_df)
plt.savefig(output_dir.joinpath("komatsu_ocean_pairplot.png"))
plt.close()
sns.pairplot(all_ocean_df)
plt.savefig(output_dir.joinpath("all_ocean_pairplot.png"))
plt.close()

def spmanr_to_annot(spmanr) -> list[list[str]]:
    annot = []
    for i, line in enumerate(spmanr.statistic):
        annot_line = []
        for k, r in enumerate(line):
            significance = ""
            pvalue = spmanr.pvalue[i, k]
            if pvalue < 0.001:
                significance = "***"
            elif pvalue < 0.01:
                significance = "**"
            elif pvalue < 0.05:
                significance = "*"
            annot_line.append(f"{r:1.3f}{significance}")
        annot.append(annot_line)

    return annot

smr = spearmanr(all_ocean_df)
fig, ax = plt.subplots(figsize=(9,9))
sns.heatmap(all_ocean_df.corr(method="spearman"), vmin=-1.0, vmax=1.0, annot=spmanr_to_annot(smr), fmt="")
fig.tight_layout()
plt.savefig(output_dir.joinpath("all_ocean_corr.png"))
plt.close()




