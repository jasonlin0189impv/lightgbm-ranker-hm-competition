from typing import Optional, Tuple
import pandas as pd


class NegativeSampling:
    def __init__(self, transaction_df: pd.DataFrame, train_inteval: int = 10):
        """
        Args:
            transaction_df (DataFrame): 交易資料表
            train_inteval (int): 取近 n 週的交易資訊當做訓練資料
        """
        self.transaction_df = transaction_df
        self.train_trans, self.valid_trans = self._train_valid_split(
            transaction_df, train_inteval
        )
        self.valid_week = transaction_df.week.max()  # 拿最後一週當作驗證

    def _train_valid_split(
        self, transaction_df: pd.DataFrame, train_interval: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """切分出驗證集, 用來產生負樣本
        * Use train_interval weeks for training
        * Use last week for validation
        """
        valid_trans = transaction_df[transaction_df.week == transaction_df.week.max()]
        train_trans = transaction_df[
            (transaction_df.week != transaction_df.week.max())
            & (transaction_df.week > transaction_df.week.max() - train_interval)
        ]
        return train_trans, valid_trans

    def create_data_with_neg_sample(
        self,
        extra_user_features: Optional[pd.DataFrame] = None,
        extra_item_features: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        # combine transaction and negative samples (candidates)
        train_trans = self.train_trans.copy()
        train_trans["purchased"] = 1

        candidates_last_purchase = self._find_last_purchase()
        candidates_bestsellers, bestsellers_previous_week = self._find_bestsellers()

        # 記錄前幾週熱銷商品給冷啟動用戶, 或是不足 12 項商品用戶使用
        self.bestsellers_previous_week = bestsellers_previous_week

        data = pd.concat(
            [train_trans, candidates_last_purchase, candidates_bestsellers]
        )
        data.purchased.fillna(0, inplace=True)
        data.drop_duplicates(["customer_id", "article_id", "week"], inplace=True)
        data = pd.merge(
            data,
            bestsellers_previous_week[["week", "article_id", "bestseller_rank"]],
            on=["week", "article_id"],
            how="left",
        )

        data = data[data.week != data.week.min()]
        data.bestseller_rank.fillna(999, inplace=True)

        if extra_item_features is not None:
            data = pd.merge(data, extra_item_features, on="article_id", how="left")
        if extra_user_features is not None:
            data = pd.merge(data, extra_user_features, on="customer_id", how="left")

        data.sort_values(["week", "customer_id"], inplace=True)
        data.reset_index(drop=True, inplace=True)

        return data

    def _find_last_purchase(self) -> pd.DataFrame:
        """找出用戶最後購買的商品, 當作負樣本 (買過之後再次購買的機率會比較低)"""
        # 找出每個使用者購物週數 (第幾週有購物)
        c2weeks = self.transaction_df.groupby("customer_id")["week"].unique()

        c2weeks2shifted_weeks = {}
        for c_id, weeks in c2weeks.items():
            c2weeks2shifted_weeks[c_id] = {}
            for i in range(weeks.shape[0] - 1):
                c2weeks2shifted_weeks[c_id][weeks[i]] = weeks[i + 1]

            c2weeks2shifted_weeks[c_id][weeks[-1]] = self.valid_week

        # 找出使用者最後購買的產品
        candidates_last_purchase = self.train_trans.copy()
        weeks = []
        for i, (c_id, week) in enumerate(
            zip(self.train_trans["customer_id"], self.train_trans["week"])
        ):
            weeks.append(c2weeks2shifted_weeks[c_id][week])

        candidates_last_purchase.week = weeks

        return candidates_last_purchase

    def _find_bestsellers(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """找出過去幾週賣最好的商品, 當作負樣本 (流行商品如果沒有被購買表示非常討厭, 會是很好的負樣本)"""
        # 過去週數商品販賣排名
        mean_price = self.train_trans.groupby(["week", "article_id"])["price"].mean()
        sales = (
            self.train_trans.groupby("week")["article_id"]
            .value_counts()
            .groupby("week")
            .rank(method="dense", ascending=False)
            .groupby("week")
            .head(12)
            .rename("bestseller_rank")
            .astype("int8")
        )

        # 過去週數商品販賣排名 & 價格
        bestsellers_previous_week = pd.merge(
            sales, mean_price, on=["week", "article_id"]
        ).reset_index()
        bestsellers_previous_week.week += 1

        unique_transactions = (
            self.train_trans.groupby(["week", "customer_id"])
            .head(1)
            .drop(columns=["article_id", "price"])
        ).copy()

        candidates_bestsellers = pd.merge(
            unique_transactions,
            bestsellers_previous_week,
            on="week",
        )

        # 找出賣最好的商品
        valid_set_transactions = unique_transactions.drop_duplicates(
            "customer_id"
        ).reset_index(drop=True)
        valid_set_transactions.week = self.valid_week

        candidates_bestsellers_valid_week = pd.merge(
            valid_set_transactions, bestsellers_previous_week, on="week"
        )

        candidates_bestsellers = pd.concat(
            [candidates_bestsellers, candidates_bestsellers_valid_week]
        )
        candidates_bestsellers.drop(columns="bestseller_rank", inplace=True)

        return candidates_bestsellers, bestsellers_previous_week
