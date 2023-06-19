from src.dataset import DATASETS
from src.models import MODELS
from src.metrics import METRICS
from src.settings import DATA_DIR
from src.evaluation import Evaluator
# from hyperparameters_original_papers.py import HPARAMS
import json
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import numpy as np

from collections import Counter

HPARAMS = {
    'instacart': {
        'num_nearest_neighbors': 900, 
        'within_decay_rate': 0.9, 
        'group_decay_rate': 0.7, 
        'alpha': 0.9, 
        'group_count': 3,
        'similarity_measure': 'euclidean'
        },
    'tafeng': {
        'num_nearest_neighbors': 300, 
        'within_decay_rate': 0.9, 
        'group_decay_rate': 0.7, 
        'alpha': 0.7, 
        'group_count': 7
        },
    'dunnhumby': {
        'num_nearest_neighbors': 900, 
        'within_decay_rate': 0.9, 
        'group_decay_rate': 0.6, 
        'alpha': 0.2, 
        'group_count': 3
        },
    'valuedshopper': {
        'num_nearest_neighbors': 300, 
        'within_decay_rate': 1, 
        'group_decay_rate': 0.6, 
        'alpha': 0.7, 
        'group_count': 7
        }
}

dataset = "tafeng"
model = "top_personal"
# metric = "recall"
batch_size = 200


cutoff_list = [10, 20]
dataset_cls = DATASETS[dataset]
model_cls = MODELS[model]

dataset_dir_name = dataset
verbose = True

data = dataset_cls(dataset_dir_name, verbose=verbose)
if dataset in ["tafeng", 'valuedshopper', 'dunnhumby']: data.make_leave_one_basket_split()
data.load_split()
print("==========")


# Start analysis of (average) basket size
# print(data.train_df)
# print('=====')
df = pd.concat([data.train_df,data.test_df, data.val_df])['basket'].tolist()
flatten = list(itertools.chain(*df))
plt.hist(flatten)
plt.savefig("lalals.png")
# print(flatten)
counter = Counter(flatten)
# print(counter)
# print(len(counter.keys()))
print(counter.most_common(50))
set1 = [x[0] for x in counter.most_common(50)]
print(set1)

# self.train_df['has_items_in_top_5_percent'] =
# data.train_df['aggregated_baskets'] = data.train_df.groupby('user_id')['basket'].apply(list)
# data.train_df['aggregated_baskets'] = data.train_df['aggregated_baskets'].apply(lambda x:[leaf for tree in x for leaf in data.train_df['aggregated_baskets']])

df_basket_counting = data.train_df[['user_id', 'basket']]
df_basket_counting['contains_top_10_percent_item'] = df_basket_counting['basket'].apply(lambda x: any([k in x for k in set1]))
df_basket_counting = df_basket_counting.drop('basket')
print(df_basket_counting)

# print(data.train_df['basket'].apply(lambda x: any([k in x for k in set1])))

assert(False)
# print(df)
# data._average_basket_size_per_user(data.train_df)

# assert False
# End analysis of (average) basket size



selected_params = HPARAMS[dataset]

# Min_freq is in the Naumov paper set between 1 and 20, and the preprocssesing to None, 'binary', or 'log'
if model == 'top_personal':
    selected_params = {
        'min_freq': 1,
        'preprocessing_popular': None,
        'preprocessing_personal': None
    }

data_object = None
with open('results/data.json', 'r') as openfile: data_object = json.load(openfile)
print(f"Data object: {data_object}")

# braycurtis was not valid
# 'manhattan', 'euclidean', 'cosine',
for s_metric in ['euclidean']:#['canberra', 'chebyshev', 'mahalanobis', 'sqeuclidean']:


    data.test_df['n_baskets_for_user'] = pd.concat([data.train_df,data.test_df, data.val_df]).groupby('user_id').count()['basket']
    #TODO: Implement mean basket size
    # data.test_df['mean_basket_size_for_user'] = pd.concat([data.train_df,data.test_df, data.val_df]).groupby('user_id')

    evaluator_test = Evaluator(dataset_df=data.test_df, cutoff_list=cutoff_list, batch_size=batch_size, verbose=verbose)

    best_vmodel = model_cls(**selected_params)
    best_vmodel.fit(dataset=data)
    performance_dct = evaluator_test.evaluate_recommender(best_vmodel)

    data_object[str([model, dataset, s_metric])]= performance_dct
    for k, v in performance_dct.items(): print(f"{k}: {v}")

with open("results/data.json", "w+") as fp: json.dump(data_object, fp, indent=4)