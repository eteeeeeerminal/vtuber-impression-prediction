import os
import sys
import pathlib
import json
import enum
import math
import yaml

import hydra
from scipy.stats import pearsonr, spearmanr
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import japanize_matplotlib

from utils.config import ModelLogConfig

sns.set(font_scale=1.4)
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['IPAexGothic']

@hydra.main(version_base=None, config_path="../config")
def main(cfg: ModelLogConfig):
    cfg.logdir = pathlib.Path(cfg.logdir)
    legend_path = pathlib.Path("data/legends").joinpath(cfg.legend)
    cfg.output = pathlib.Path(cfg.output)
    os.makedirs(cfg.output, exist_ok=True)

    def csv_to_df(
        path: pathlib.Path,
        keys: list[str] = ["step", "train_loss", "valid_loss"]
    ) -> pd.DataFrame:
        df = pd.read_csv(path)[keys]
        return df

    df_list: list[pd.DataFrame] = [
        csv_to_df(cfg.logdir.joinpath(f"{cfg.logname}-{i}").joinpath("metrics.csv"))
        for i in range(cfg.run_n)
    ]

    def df_min(i: int, key="valid_loss") -> float:
        _min = df_list[i][key].min()
        if math.isnan(_min):
            return 1000
        return _min

    # valid_loss が最良のもの5つ抽出
    df_keys = sorted(
        list(range(cfg.run_n)),
        key=df_min
    )[:5]
    print(df_keys)

    # train_loss が最良のもの5つ抽出
    print(sorted(
        list(range(cfg.run_n)),
        key=lambda i: df_min(i, "train_loss")
    )[:5])

    # legend の読み込み
    def read_legend(path: pathlib.Path) -> str:
        with open(path, "r", encoding="utf-8") as f:
            legend_yaml = yaml.safe_load(f)
            legend = ",    ".join([
                param.split("=")[1] for param in legend_yaml
            ])
        return legend

    train_mat = []
    valid_mat = []
    for i in df_keys:
        for row in df_list[i].iterrows():
            row = row[1]
            step = row["step"]
            train_loss = row["train_loss"]
            valid_loss = row["valid_loss"]
            legend = read_legend(legend_path.joinpath(f"{i}/.hydra/overrides.yaml"))
            if not np.isnan(train_loss):
                train_mat.append([step, legend, train_loss])

            if not np.isnan(valid_loss):
                valid_mat.append([step, legend, valid_loss])

    train_df = pd.DataFrame(
        train_mat,
        columns=["step", "legend", "train_loss"]
    )
    valid_df = pd.DataFrame(
        valid_mat,
        columns=["step", "legend", "valid_loss"]
    )

    def plot(df: pd.DataFrame, name: str):
        fig, axis = plt.subplots(figsize=(13,6))
        axis.axes.set_ylim(cfg.y_min, cfg.y_max)
        p = sns.lineplot(df.pivot("step", "legend", name), ax=axis)
        p.set_ylabel(name)
        axis.legend(
            loc='upper left', bbox_to_anchor=(1.05,1),
            title=cfg.legend_title.rjust(len(cfg.legend_title)+10, ' '),
            alignment="left"
        )
        axis.axes.set_ylim(cfg.y_min, cfg.y_max)
        fig.tight_layout()
        plt.savefig(cfg.output.joinpath(name))
        plt.close()

    plot(train_df, "train_loss")
    plot(valid_df, "valid_loss")

if __name__ == "__main__":
    main()
