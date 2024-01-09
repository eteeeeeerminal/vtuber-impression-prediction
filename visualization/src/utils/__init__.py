import numpy as np
import pandas as pd

def cronback_alpha(df: pd.DataFrame) -> float:
    n_item = len(df.columns)
    item_var = sum(df.var())
    total_var = df.sum(axis=1).var()
    alpha = n_item / (n_item - 1) * (1 - (item_var / total_var))
    return alpha
