import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import List, Union


class ItemFeatures(ABC):
    @abstractmethod
    def get(self, *args, **kwargs) -> pd.DataFrame:
        """
        article_id -> features
        """
        pass


class CategoryTransform(ItemFeatures):
    """
    factorize all articles columns
    """

    def __init__(self, articles_df: pd.DataFrame):
        self.articles_df = articles_df

    def get(self):
        self.__feature_columns = list(
            filter(lambda x: "name" in x, self.articles_df.columns)
        )[1:]
        filtered_articles = self.articles_df[self.__feature_columns]
        filtered_articles = filtered_articles.apply(lambda x: pd.factorize(x)[0])
        filtered_articles["article_id"] = self.articles_df["article_id"]

        features = filtered_articles.set_index("article_id").astype("int8")
        return features

    def get_columns(self):
        return self.__feature_columns


class AggrTransform(ItemFeatures):
    """
    aggregation transactions features : mean, max and etc...
    """

    def __init__(self, articles_df: pd.DataFrame, transactions_df: pd.DataFrame):
        self.articles_df = articles_df
        self.transactions_df = transactions_df

    def get(self):
        stats = self._get_stats()
        return stats

    def _get_stats(self):
        transactions_more = self.transactions_df.merge(
            self.articles_df, on=("article_id")
        )
        grouped = transactions_more.groupby("article_id")

        counts = (
            grouped["article_id"]
            .count()
            .to_frame()
            .rename(columns={"article_id": "count"})
            .astype("int16")
            .reset_index()
            .set_index("article_id")
        )
        sums = (
            grouped["price"]
            .sum()
            .to_frame()
            .astype("float32")
            .rename(columns={"price": "sum_price"})
        )
        means = (
            grouped["price"]
            .mean()
            .to_frame()
            .astype("float32")
            .rename(columns={"price": "mean_price"})
        )
        mins = (
            grouped["price"]
            .min()
            .to_frame()
            .astype("float32")
            .rename(columns={"price": "min_price"})
        )
        maxs = (
            grouped["price"]
            .max()
            .to_frame()
            .astype("float32")
            .rename(columns={"price": "max_price"})
        )

        output_df = (
            counts.merge(sums, on=("article_id"))
            .merge(means, on=("article_id"))
            .merge(mins, on=("article_id"))
            .merge(maxs, on=("article_id"))
        )
        return output_df


class TopTransforms(ItemFeatures):
    """
    whether category appears in top categories
    """

    def __init__(self, articles_df: pd.DataFrame, topk=3):
        self.articles_df = articles_df
        self.topk = topk

    def get(self):
        name_cols = list(filter(lambda x: "name" in x, self.articles_df.columns))

        value_counts_dict = self._get_value_counts(name_cols)
        value_counts_dict = {
            f"{k}_{self.topk}": self.articles_df[k].isin(v).astype("int8")
            for k, v in value_counts_dict.items()
        }

        output_df = self.articles_df.assign(**value_counts_dict)
        output_df = output_df[
            ["article_id"] + list(value_counts_dict.keys())
        ].set_index("article_id")
        return output_df

    def _get_value_counts(self, name_cols: List[str]):
        value_counts = self.articles_df[name_cols].apply(pd.Series.value_counts)
        get_index = lambda x: value_counts.sort_values(x, ascending=False)[x][
            : self.topk
        ].index
        value_counts_dict = dict(zip(name_cols, map(lambda x: get_index(x), name_cols)))

        return value_counts_dict


class ItemFeaturesCollector:
    @staticmethod
    def collect(
        features: Union[List[ItemFeatures], List[str]], **kwargs
    ) -> pd.DataFrame:
        output_df = None

        for feature in tqdm(features):
            if isinstance(feature, ItemFeatures):
                feature_out = feature.get(**kwargs)
            if isinstance(feature, str):
                try:
                    feature_out = pd.read_csv(feature)
                except:
                    feature_out = pd.read_parquet(feature)

            if output_df is None:
                output_df = feature_out
            else:
                output_df = output_df.merge(feature_out, on=("article_id"))
        return output_df
