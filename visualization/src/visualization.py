import re
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

from dataset import DatasetPack
from utils import cronback_alpha
from word_cloud import WordCloud, normalize_word

default_font_scale = 1.4
def set_font(font_size: float = default_font_scale):
    sns.set_theme()
    sns.set(font_scale=font_size)
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['IPAexGothic']
set_font()

class Mode(enum.Enum):
    all = "all"
    pop = "pop"
    word = "word"
    word_list = "word_list"
    stat = "stat"
    onom_per = "onom_per"
    tipij_per = "tipij_per"
    onom_tipij = "onom_tipij"
    ml = "ml"

parser = argparse.ArgumentParser()
parser.add_argument("-m", default="all", type=str) # mode
parser.add_argument("-i", default="../preprocess-for-ml/data/labeled-dataset/labeled-dataset.json", type=str)
parser.add_argument("-v", default="./data/merged.json", type=str) # VTuber データセット
parser.add_argument("-o", default="./output/annotation", type=str)
args = parser.parse_args()

mode = str(args.m)
dataset_path = pathlib.Path(args.i)
vtuber_data_path = pathlib.Path(args.v)
output_dir = pathlib.Path(args.o)

os.makedirs(output_dir, exist_ok=True)

dataset = DatasetPack.from_json(dataset_path)
with open(vtuber_data_path, "r", encoding="utf-8") as f:
    vtuber_data = json.load(f)

annotated_vtuber_ids = set([
    data.origin.input.youtube_id for data in dataset.dataset
])
annotated_vtuber_data = list(filter(lambda x: x["vtuber_id"] in annotated_vtuber_ids, vtuber_data))

exec_all = mode == Mode.all

# VTuber 聞いたことあるかどうかの割合
if exec_all or Mode.pop.value in mode:
    pop_output_dir = output_dir.joinpath(Mode.pop.value)
    os.makedirs(pop_output_dir, exist_ok=True)

    popularity_label = "このVTuberをすでに知っていましたか?"
    popularity = [["", data.origin.label.already_know] for data in dataset.dataset]
    popularity = pd.DataFrame(popularity, columns=["", popularity_label])

    print(popularity.value_counts())

    d = 0.04
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, sharey=True, figsize=(12,5))
    sns.countplot(popularity, y=popularity_label, ax=ax1)
    sns.countplot(popularity, y=popularity_label, ax=ax2)
    ax1.set_xlim(0, 20)
    kwargs = dict(transform=ax1.transAxes, color='gray', linestyle='--', lw=4, clip_on=False)
    ax1.plot((1, 1), (-d,1+d) , **kwargs)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.set_ylabel("")
    ax1.set_xlabel("")

    ax2.set_xlim(770, 790)
    kwargs = dict(transform=ax2.transAxes, color='gray', linestyle='--', lw=4, clip_on=False)
    ax2.plot((0, 0), (-d,1+d) , **kwargs)
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.set_ylabel("")
    ax2.set_xlabel("")

    # plt.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle(popularity_label, y=0.95)
    fig.supxlabel("回答数", x=0.55, y=0.06, ha="left")

    fig.tight_layout()
    plt.savefig(pop_output_dir.joinpath("popularity.png"))
    plt.close()

    # チャンネル登録者などのグラフ化
    #"""
    youtube_info = [[data["youtube"]["subscriber_count"], data["youtube"]["view_count"]]
        for data in annotated_vtuber_data
    ]
    subscribe_label = "登録者数"
    view_count = "総視聴回数"
    youtube_df = pd.DataFrame(youtube_info, columns=[subscribe_label, view_count])
    fig, ax = plt.subplots()
    sns.histplot(youtube_df[subscribe_label], log_scale=10, ax=ax)
    ax.set_ylabel("チャンネル数")
    fig.tight_layout()
    plt.savefig(pop_output_dir.joinpath("youtube.png"))
    plt.close()
    # """

def word_counter_to_table(word_counter: Counter, include_count: bool = True) -> str:
    ret_str = ""
    for i, (word, count) in enumerate(word_counter.most_common()):
        if include_count and count > 1:
            ret_str += f"{word} & {count}\n"
        else:
            ret_str += f"{word}\n"
    return ret_str

if exec_all or Mode.word_list.value in mode:
    wl_output_dir = output_dir.joinpath("word")
    os.makedirs(wl_output_dir, exist_ok=True)

    first_onoms = map(lambda x: x.origin.label.first_onom, dataset.dataset)
    first_onoms = list(map(normalize_word, first_onoms))
    first_onom_counter = Counter(first_onoms)
    other_onoms = map(lambda x: re.split("[，、 　]", x.origin.label.other_onom), dataset.dataset)
    other_onoms = sum(other_onoms, [])
    other_onoms = filter(lambda x: x and x not in first_onom_counter, other_onoms)
    other_onoms = list(map(normalize_word, other_onoms))
    other_onom_counter = Counter(other_onoms)

    with open(wl_output_dir.joinpath("word_list.txt"), "w", encoding="utf-8") as f:
        f.write(word_counter_to_table(first_onom_counter))
        f.write("\n\n")
        f.write(word_counter_to_table(other_onom_counter, include_count=False))


# ワードクラウドの作成
if exec_all or Mode.word.value in mode:
    w_output_dir = output_dir.joinpath("word")
    os.makedirs(w_output_dir, exist_ok=True)

    wordcloud = WordCloud()
    ## メインのオノマトペのみ
    wordcloud.generate_first_onom_cloud(dataset, w_output_dir.joinpath("cloud-first-onom.png"))

    ## 全オノマトペ
    wordcloud.generate_all_onom_cloud(dataset, w_output_dir.joinpath("cloud-all-onom.png"))

    ## その他の印象
    wordcloud.generate_impression_cloud(dataset, w_output_dir.joinpath("cloud-impressions.png"))
    # """

# オノマトペとTIPIJのベクトルの統計量を見てみる
def generate_boxplot(df: pd.DataFrame, name: str, output_dir: pathlib.Path):
    fig, ax = plt.subplots()
    sns.boxenplot(df)
    fig.tight_layout()
    plt.savefig(output_dir.joinpath(f"{name}_boxplots.png"))
    plt.close()

    fig, ax = plt.subplots()
    df = df.describe()[1:3]
    sns.boxenplot(df.T)
    fig.tight_layout()
    plt.savefig(output_dir.joinpath(f"{name}_stats_boxplots.png"))
    plt.close()

def generate_violinplot(df: pd.DataFrame, name: str, output_dir: pathlib.Path, n: int = 10):
    columns_n = len(df.columns)
    for i in range(0, int(columns_n / n)+1):
        if columns_n == i*n:
            break
        fig, ax = plt.subplots(figsize=(14,6))
        sns.violinplot(df.iloc[:, i*n: (i+1)*n])
        fig.tight_layout()
        plt.savefig(output_dir.joinpath(f"{name}_violinplots_{i}.png"))
        plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    df = df.describe()[1:3]
    sns.violinplot(df.T)
    fig.tight_layout()
    plt.savefig(output_dir.joinpath(f"{name}_stats_violinplots.png"))
    plt.close()

def scales_to_columns(scales: list[str]) -> list[str]:
    columns = map(lambda scale: scale.split(","), scales)
    columns = map(lambda s: "\n".join(["+: "+s[1], "-: "+s[0]]), columns)
    return list(columns)

def tipij_scales_to_columns(scales: list[str]) -> list[str]:
    return [s.replace("、", "、\n") for s in scales]

# バイオリンプロット
if exec_all or Mode.stat.value in mode:
    stat_output_dir = output_dir.joinpath("stat")
    os.makedirs(stat_output_dir, exist_ok=True)

    # オノマトペ印象
    onom_imp_vec_df = pd.DataFrame(
        [data.onom.impression for data in dataset.dataset],
        columns=scales_to_columns(dataset.onom_imp_scale)
    )
    # generate_boxplot(onom_imp_vec_df, "onom_imp")
    generate_violinplot(onom_imp_vec_df, "onom_imp", stat_output_dir, 5)

    # オノマトペ性格
    onom_personality_vec_df = pd.DataFrame(
        [data.onom.personality for data in dataset.dataset],
        columns=scales_to_columns(dataset.onom_personality_scale)
    )
    # generate_boxplot(onom_personality_vec_df, "onom_personality")
    generate_violinplot(onom_personality_vec_df, "onom_personality", stat_output_dir, 5)

    # TIPIJ
    tipij_df = pd.DataFrame(
        [data.tipij.personality for data in dataset.dataset],
        columns=tipij_scales_to_columns(dataset.tipij_scale)
    )
    # generate_boxplot(tipij_df, "tipij")
    generate_violinplot(tipij_df, "tipij", stat_output_dir, 5)

ocean_to_jp = {
    "O": "O (開放性)",
    "C": "C (誠実性)",
    "E": "E (外向性)",
    "A": "A (調和性)",
    "N": "N (情緒不安定性)"
}

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

if exec_all or Mode.onom_per.value in mode:
    onom_per_output_dir = output_dir.joinpath("onom-per")
    os.makedirs(onom_per_output_dir, exist_ok=True)

    set_font(8.0)
    onom_personality_vec_df = pd.DataFrame(
        [data.onom.personality for data in dataset.dataset],
        columns=dataset.onom_personality_scale
    )
    onom_per_corr = onom_personality_vec_df.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(50,50))
    sns.heatmap(onom_per_corr, vmin=-1.0, vmax=1.0, xticklabels=False, yticklabels=False)
    fig.tight_layout()
    plt.savefig(onom_per_output_dir.joinpath("onom_per_corr.png"))
    plt.close()
    set_font()

    print(f"E: {cronback_alpha(onom_personality_vec_df.iloc[:,  0:10])}")
    print(f"A: {cronback_alpha(onom_personality_vec_df.iloc[:, 10:20])}")
    print(f"C: {cronback_alpha(onom_personality_vec_df.iloc[:, 20:30])}")
    print(f"N: {cronback_alpha(onom_personality_vec_df.iloc[:, 30:40])}")
    print(f"O: {cronback_alpha(onom_personality_vec_df.iloc[:, 40:50])}")

    onom_ocean_scale = [ocean_to_jp[scale] for scale in dataset.onom_personality_summarized_scale]
    onom_ocean = pd.DataFrame(
        [data.onom.personality_summarized for data in dataset.dataset],
        columns=onom_ocean_scale
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(onom_ocean)
    fig.tight_layout()
    plt.savefig(onom_per_output_dir.joinpath("onom_ocean_violinplots.png"))
    plt.close()
    sns.pairplot(onom_ocean)
    plt.savefig(onom_per_output_dir.joinpath("onom_ocean_pairplot.png"))
    plt.close()

    smr = spearmanr(onom_ocean)
    fig, ax = plt.subplots(figsize=(9,9))
    sns.heatmap(onom_ocean.corr(method="spearman"), vmin=-1.0, vmax=1.0, annot=spmanr_to_annot(smr), fmt="")
    fig.tight_layout()
    plt.savefig(onom_per_output_dir.joinpath("onom_ocean_corr.png"))
    plt.close()


if exec_all or Mode.tipij_per.value in mode:
    tipij_per_output_dir = output_dir.joinpath("tipij-per")
    os.makedirs(tipij_per_output_dir, exist_ok=True)

    tipij_df = pd.DataFrame(
        [data.tipij.personality for data in dataset.dataset],
        columns=tipij_scales_to_columns(dataset.tipij_scale)
    )
    smr = spearmanr(tipij_df)
    tipij_corr = tipij_df.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(16, 16))
    sns.heatmap(tipij_corr, vmin=-1.0, vmax=1.0, annot=spmanr_to_annot(smr), fmt="")
    fig.tight_layout()
    plt.savefig(tipij_per_output_dir.joinpath("tipij_corr.png"))
    plt.close()
    # sns.pairplot(tipij_df, kind="hist")
    # plt.savefig(tipij_per_output_dir.joinpath("tipij_pairplot.png"))
    # plt.close()

    tipij_ocean_scale = [ocean_to_jp[scale] for scale in dataset.tipij_summarized_scale]
    tipij_ocean = pd.DataFrame(
        [data.tipij.personality_summarized for data in dataset.dataset],
        columns=tipij_ocean_scale
    )
    fig, ax = plt.subplots()
    sns.violinplot(tipij_ocean)
    fig.tight_layout()
    plt.savefig(tipij_per_output_dir.joinpath("tipij_ocean_violinplots.png"))
    plt.close()
    sns.pairplot(tipij_ocean, kind="hist", plot_kws={"bins": 12}, diag_kws={"bins": 12})
    plt.savefig(tipij_per_output_dir.joinpath("tipij_ocean_pairplot.png"))
    plt.close()

    smr = spearmanr(tipij_ocean)
    fig, ax = plt.subplots(figsize=(9,9))
    sns.heatmap(tipij_ocean.corr(method="spearman"), vmin=-1.0, vmax=1.0, annot=spmanr_to_annot(smr), fmt="")
    fig.tight_layout()
    plt.savefig(tipij_per_output_dir.joinpath("tipij_ocean_corr.png"))
    plt.close()


if exec_all or Mode.onom_tipij.value in mode:
    onom_tipij_output_dir = output_dir.joinpath("onom-tipij")
    os.makedirs(onom_tipij_output_dir, exist_ok=True)

    onom_ocean_scale = [scale + "_onom" for scale in dataset.onom_personality_summarized_scale]
    tipij_ocean_scale = [scale + "_tipij" for scale in dataset.tipij_summarized_scale]
    all_ocean = pd.DataFrame(
        [data.onom.personality_summarized  + data.tipij.personality_summarized for data in dataset.dataset],
        columns = onom_ocean_scale + tipij_ocean_scale
    )

    sns.pairplot(all_ocean, x_vars=tipij_ocean_scale, y_vars=onom_ocean_scale, kind="hist", plot_kws={"bins": 12}, diag_kws={"bins": 12})
    plt.savefig(onom_tipij_output_dir.joinpath("onom_tipij_pairplot.png"))
    plt.close()

    smr = spearmanr(all_ocean)
    fig, ax = plt.subplots(figsize=(14,14))
    sns.heatmap(all_ocean.corr(method="spearman"), vmin=-1.0, vmax=1.0, annot=spmanr_to_annot(smr), fmt="")
    fig.tight_layout()
    plt.savefig(onom_tipij_output_dir.joinpath("onom_tipij_corr.png"))
    plt.close()



    # 印象評価値の相関でも見ますか
    """
    imp_df = pd.DataFrame(
        [data.onom.impression for data in dataset.dataset],
        columns=dataset.onom_imp_scale
    )
    onom_per_scale = np.array(dataset.onom_imp_scale)
    sns.pairplot(imp_df, vars=onom_per_scale[[0, 10, 20, 30, 40]])
    imp_df.corr().to_csv("impression.csv", sep="\t")
    plt.savefig(output_dir.joinpath("onom_imp_pair.png"))
    # """

    # 最後に全尺度? は多いから、印象全部+OCEAN*2で相関係数
    """
    summary_df = pd.DataFrame(
        [data.onom.impression + data.onom.personality_summarized + data.tipij.personality_summarized for data in dataset.dataset],
        columns=dataset.onom_imp_scale + onom_ocean_scale + tipij_ocean_scale
    )
    plt.figure(figsize=(50,50))
    sns.heatmap(summary_df.corr(), annot=True, fmt="1.3f")
    plt.savefig(output_dir.joinpath("summary_heatmap.png"))
    # """

# 全尺度で、0以上の値の割合を見る
if exec_all or Mode.ml.value in mode:
    all_df = pd.DataFrame(
        [data.onom.impression + data.onom.personality + data.tipij.personality for data in dataset.dataset],
        columns=dataset.onom_imp_scale + dataset.onom_personality_scale + dataset.tipij_scale
    )

    df_positive = (all_df > 0.0)
    result = df_positive.apply(lambda x: pd.value_counts(x, normalize=True))
    print(f"Acc: {result.max().mean()}")
    result.max().to_csv("acc_baseline.csv")

    # 平均値だして色々してみる
    print(f"全部0のときの2乗誤差: {((all_df)**2).mean().mean()}")
    print(f"全部平均値のときの2乗誤差: {((all_df - all_df.mean())**2).mean().mean()}")
    ((all_df - all_df.mean())**2).mean().to_csv("loss_baseline.csv")

    print("以下値域を-1~1 → 0~1へ")
    normalized = (all_df + 1.0) / 2.0
    print(f"全部0.5のときの2乗誤差: {((normalized - 0.5)**2).mean().mean()}")
    print(f"全部平均値のときの2乗誤差: {((normalized - normalized.mean())**2).mean().mean()}")
