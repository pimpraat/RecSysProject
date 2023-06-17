import os

import pandas as pd

from .base import NBRDatasetBase


class TianchiDataset(NBRDatasetBase):
    def __init__(
        self,
        dataset_folder_name: str = "tianchi",
        min_baskets_per_user: int = 3,
        min_items_per_user: int = 0,
        min_users_per_item: int = 1,
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

        transaction_data_path = os.path.join(self.raw_path, "tianchi_fresh_comp_train_user.csv")
        df = pd.read_csv(transaction_data_path)

        df = df[df['behavior_type'] == 4]

        df = df.drop(columns=['behavior_type', 'user_geohash', 'item_category'])

        df.insert(loc=0, column='basket_id',
                        value=df.set_index(['user_id', 'time']).index.factorize()[0] + 1)

        df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H')

        df = df.rename(
            columns={'time': 'timestamp'}
        )

        df = df.drop_duplicates()

        return df