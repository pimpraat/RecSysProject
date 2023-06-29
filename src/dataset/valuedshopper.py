import os

import pandas as pd

from .base import NBRDatasetBase


class VSDataset(NBRDatasetBase):
    def __init__(
            self,
            dataset_folder_name: str = "valuedshopper",
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

        transaction_data_path1 = os.path.join(self.raw_path, "VS_future_order.csv")
        transaction_data_path2 = os.path.join(self.raw_path, "VS_history_order.csv")
        df1 = pd.read_csv(transaction_data_path1)
        df2 = pd.read_csv(transaction_data_path2)

        df = pd.concat([df1, df2], ignore_index=True)

        df['timestamp'] = pd.to_datetime(df['ORDER_NUMBER'])
        df = df.rename(
            columns={'CUSTOMER_ID': 'user_id', 'ORDER_NUMBER': 'basket_id', 'MATERIAL_NUMBER': 'item_id'}
        )

        df = df.drop_duplicates()

        return df
