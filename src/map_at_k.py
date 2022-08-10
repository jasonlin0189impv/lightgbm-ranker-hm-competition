"""
Reference: https://www.kaggle.com/code/mayukh18/time-decaying-popularity-benchmark-0-0216/notebook
GitHub: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
"""

from typing import List
import numpy as np
import pandas as pd


def apk(actual, predicted, k=12):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        # not in 是確認沒有重複出現？
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(
    measure_df: pd.DataFrame,
    pred_col: str = "prediction",
    ground_true_col: str = "ground_true",
    k=12,
):
    apks = []
    pred_list: List[List[str]] = measure_df[pred_col].to_list()
    ground_true_list: List[List[str]] = measure_df[ground_true_col].to_list()

    for pred, g_true in zip(pred_list, ground_true_list):
        apks.append(apk(g_true, pred, k=12))

    return np.mean(apks)
