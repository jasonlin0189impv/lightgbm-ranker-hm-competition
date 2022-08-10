from tokenize import group
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from tqdm import tqdm
from typing import List, Union


class UserFeatures(ABC):
    @abstractmethod
    def get(self) -> pd.DataFrame:
        """
        customer_id -> features
        """
        pass


class AggrFeatures(UserFeatures):
    """
    basic aggregation features(min, max, mean and etc...)
    """

    def __init__(self, transactions_df):
        self.groupby_df = transactions_df.groupby("customer_id", as_index=False)

    def get(self):
        output_df = (
            self.groupby_df["price"]
            .agg(
                {
                    "mean_transactions": "mean",
                    "max_transactions": "max",
                    "min_transactions": "min",
                    "median_transactions": "median",
                    "sum_transactions": "sum",
                }
            )
            .set_index("customer_id")
            .astype("float32")
        )
        output_df["max_minus_min_transactions"] = (
            output_df["max_transactions"] - output_df["min_transactions"]
        )

        return output_df


class CountFeatures(UserFeatures):
    """
    basic features connected with transactions
    """

    def __init__(self, transactions_df, topk=10):
        self.transactions_df = transactions_df
        self.topk = topk

    def get(self):
        grouped = self.transactions_df.groupby("customer_id", as_index=False)

        # count number of transaction in online & offline
        count_article_online_and_offline = (
            self.transactions_df.groupby(["customer_id", "sales_channel_id"])
            .size()
            .unstack(fill_value=0)
        ).rename(columns={1: "count_store_articles", 2: "count_online_articles"})

        # number of transactions, number of online articles,
        # number of transactions bigger than mean price of transactions
        basic_info = (
            grouped.agg(
                n_transactions=("article_id", "count"),
                n_unique_articles=("article_id", "nunique"),
            )
            .set_index("customer_id")
            .astype("int8")
        )

        # number of transactions that are in top
        topk_articles = (
            self.transactions_df["article_id"].value_counts()[: self.topk].index
        )
        num_transaction_in_top_n = self.transactions_df.copy()
        for i, k in enumerate(topk_articles):
            num_transaction_in_top_n[f"top_aritcle_{i}"] = k
            num_transaction_in_top_n[f"count_top_aritcle_{i}"] = (
                num_transaction_in_top_n["article_id"]
                == num_transaction_in_top_n[f"top_aritcle_{i}"]
            )
        num_transaction_in_top_n = (
            num_transaction_in_top_n.groupby("customer_id", as_index=False)
            .agg({f"count_top_aritcle_{i}": "sum" for i in range(self.topk)})
            .set_index("customer_id")
            .astype("int8")
        )

        # number of transaction price greater than mean
        mean_info = grouped.agg(mean_value=("price", "mean"))
        n_transactions_price_greater_than_mean = self.transactions_df.copy()
        n_transactions_price_greater_than_mean = (
            n_transactions_price_greater_than_mean.merge(
                mean_info, on="customer_id", how="left"
            )
        )
        n_transactions_price_greater_than_mean["compare"] = (
            n_transactions_price_greater_than_mean["price"]
            > n_transactions_price_greater_than_mean["mean_value"]
        )
        n_transactions_price_greater_than_mean = (
            n_transactions_price_greater_than_mean.groupby("customer_id").agg(
                n_transactions_price_greater_than_mean=("compare", "sum")
            )
        )

        output_df = (
            count_article_online_and_offline.merge(basic_info, on=("customer_id"))
            .merge(num_transaction_in_top_n, on=("customer_id"))
            .merge(n_transactions_price_greater_than_mean, on=("customer_id"))
        )
        return output_df


class CustomerFeatures(UserFeatures):
    """
    All columns from customers dataframe
    """

    def __init__(self, customers_df, use_age=True):
        self.customers_df = self._prepare_customers(customers_df, use_age)

    def _prepare_customers(self, customers_df, use_age):
        customers_df["FN"] = customers_df["FN"].fillna(0).astype("int8")
        customers_df["Active"] = customers_df["Active"].fillna(0).astype("int8")
        customers_df["club_member_status"] = customers_df["club_member_status"].fillna(
            "UNKNOWN"
        )
        if use_age:
            customers_df["age"] = (
                customers_df["age"]
                .fillna(customers_df["age"].mean())
                .astype("int8")  # can change mean to mode
            )
        customers_df["fashion_news_frequency"] = (
            customers_df["fashion_news_frequency"]
            .replace("None", "NONE")
            .replace(np.nan, "NONE")
        )
        return customers_df

    def get(self):
        output = self.customers_df[
            filter(lambda x: x != "postal_code", self.customers_df.columns)
        ].set_index("customer_id")
        return output


class ArticlesFeatures(UserFeatures):
    """
    returns article features: whether category appears in top categories
    """

    def __init__(self, transactions_df, articles, topk=10):
        self.merged_df = transactions_df.merge(articles, on=("article_id"))
        self.articles = articles
        self.topk = topk

    def get(self):
        output_df = None

        for col in tqdm(self.articles.columns, desc="extracting features"):
            if "name" in col:
                if output_df is None:
                    output_df = self.aggregate_topk(self.merged_df, col, self.topk)
                else:
                    intermediate_out = self.aggregate_topk(
                        self.merged_df, col, self.topk
                    )
                    output_df = output_df.merge(intermediate_out, on=("customer_id"))
        return output_df

    def return_value_counts(self, df, column_name, k):
        value_counts = df[column_name].value_counts()[:k].index
        value_counts = list(map(lambda x: x[1], value_counts))
        return value_counts

    def aggregate_topk(self, merged_df, column_name, k):
        grouped_df_indx = merged_df.groupby("customer_id")

        topk_values = self.return_value_counts(grouped_df_indx, column_name, k)
        # how many transactions appears in top category(column)
        article_feature_in_n_top_k = merged_df.copy()
        for i, i_name in enumerate(topk_values):
            article_feature_in_n_top_k[f"top_{column_name}_{i}"] = i_name
            article_feature_in_n_top_k[f"count_top_{column_name}_{i}"] = (
                article_feature_in_n_top_k[column_name]
                == article_feature_in_n_top_k[f"top_{column_name}_{i}"]
            )

        output = (
            article_feature_in_n_top_k.groupby("customer_id", as_index=False)
            .agg(
                {f"count_top_{column_name}_{i}": "sum" for i in range(len(topk_values))}
            )
            .set_index("customer_id")
            .astype("int16")
        )

        return output


class UserFeaturesCollector:
    """
    collect all features and aggregate them
    """

    @staticmethod
    def collect(
        features: Union[List[UserFeatures], List[str]], **kwargs
    ) -> pd.DataFrame:
        output_df = None

        for feature in tqdm(features):
            if isinstance(feature, UserFeatures):
                feature_out = feature.get(**kwargs)
            if isinstance(feature, str):
                try:
                    feature_out = pd.read_csv(feature)
                except:
                    feature_out = pd.read_parquet(feature)

            if output_df is None:
                output_df = feature_out
            else:
                output_df = output_df.merge(feature_out, on=("customer_id"))
        return output_df
