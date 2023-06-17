import os

import pandas as pd

from .base import NBRDatasetBase


class TmallDataset(NBRDatasetBase):
    def __init__(
        self,
        dataset_folder_name: str = "tmall",
        min_baskets_per_user: int = 3,
        min_items_per_user: int = 0,
        min_users_per_item: int = 5,
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

        transaction_data_path = os.path.join(self.raw_path, "ijcai2016_taobao.csv")
        df = pd.read_csv(transaction_data_path)

        df = df[df['act_ID'] == 1]

        df = df.drop(columns=['act_ID', 'cat_ID', 'sel_ID'])

        df.insert(loc=0, column='basket_id',
                        value=df.set_index(['use_ID', 'time']).index.factorize()[0] + 1)

        df['time'] = pd.to_datetime(df['time'], format='%Y%m%d')

        df = df.rename(
            columns={"use_ID": "user_id", "ite_ID": "item_id", 'time': 'timestamp'}
        )

        df = df.drop_duplicates()

        item_counts = df.groupby(['user_id', 'basket_id'])['item_id'].count().reset_index().rename(
            {'item_id': 'item_count'}, axis=1)

        df = df.merge(item_counts, on=['user_id', 'basket_id'], how='left')

        df = df[df.item_count >= 4]

        df.drop(columns=['item_count'], inplace=True)

        return df