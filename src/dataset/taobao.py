import os

import pandas as pd

from .base import NBRDatasetBase


class TaobaoDataset(NBRDatasetBase):
    def __init__(
            self,
            dataset_folder_name: str = "taobao",
            min_baskets_per_user: int = 3,
            min_items_per_user: int = 0,
            min_users_per_item: int = 0,
            verbose=False,
    ):
        super().__init__(
            dataset_folder_name,
            min_baskets_per_user=min_baskets_per_user,
            min_items_per_user=min_items_per_user,
            min_users_per_item=min_users_per_item,
            verbose=verbose,
        )

    def _preprocess(self) -> pd.DataFrame:
        transaction_data_path = os.path.join(self.raw_path, "taobao.csv")
        df = pd.read_csv(transaction_data_path, header=None,
                            names=['user_id', 'item_id', 'category_id', 'action', 'timestamp'])

        df = df[df['action'] == 'buy']

        df = df.drop(columns=['category_id', 'action'])

        df.insert(loc=0, column='basket_id',
                  value=df.set_index(['user_id', 'timestamp']).index.factorize()[0]+1)

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        df = df.drop_duplicates()

        item_counts = df.groupby(['user_id', 'basket_id'])['item_id'].count().reset_index().rename(
            {'item_id': 'item_count'}, axis=1)

        df = df.merge(item_counts, on=['user_id', 'basket_id'], how='left')

        df = df[df.item_count >= 2]

        df.drop(columns=['item_count'], inplace=True)

        return df