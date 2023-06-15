from src.dataset import DATASETS
from src.models import MODELS
from src.metrics import METRICS
from src.settings import DATA_DIR
from src.evaluation import Evaluator
# from hyperparameters_original_papers.py import HPARAMS
import json

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

dataset = "dunnhumby"
model = "top_personal"
# metric = "recall"
batch_size = 200


cutoff_list = [10, 20]
dataset_cls = DATASETS[dataset]
model_cls = MODELS[model]

dataset_dir_name = dataset
verbose = True

data = dataset_cls(dataset_dir_name, verbose=verbose)
if dataset in ["tafeng", 'valuedshopper']: data.make_leave_one_basket_split()
data.load_split()
print("==========")


# Start analysis of (average) basket size
print(data.train_df)


# End analysis of (average) basket size

r10, r20, ndcg10, ncdg20 = [], [], [], []



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
    evaluator_test = Evaluator(dataset_df=data.test_df, cutoff_list=cutoff_list, batch_size=batch_size, verbose=verbose)

    best_vmodel = model_cls(**selected_params)

    best_vmodel.fit(dataset=data)

    performance_dct = evaluator_test.evaluate_recommender(best_vmodel)

    # key = {
    #     'model': model,
    #     'dataset': dataset,
    #     'sim_measure': s_metric
    #     }

    data_object[str([model, dataset, s_metric])]= performance_dct
    for k, v in performance_dct.items(): print(f"{k}: {v}")

with open("results/data.json", "w+") as fp: json.dump(data_object, fp, indent=4)

# for datapercentage in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
#     data.restrict_dataset(column_to_restrict='user_id', perc_ids_to_keep=datapercentage)


#     for s_metric in ['euclidean']:#['canberra', 'chebyshev', 'mahalanobis', 'sqeuclidean']:
#         evaluator_test = Evaluator(dataset_df=data.test_df, cutoff_list=cutoff_list, batch_size=batch_size, verbose=verbose)
#         vparams = {
#             'num_nearest_neighbors': 900, 
#             'within_decay_rate': 0.9, 
#             'group_decay_rate': 0.7, 
#             'alpha': 0.9, 
#             'group_count': 3,
#             'similarity_measure': s_metric
#             }

#     best_vmodel = model_cls(**vparams)

#     best_vmodel.fit(dataset=data)

#     performance_dct = evaluator_test.evaluate_recommender(best_vmodel)
#     print(f" Performance using the {s_metric} as similarity measure: {performance_dct}")
#     r10.append(performance_dct['recall@010'])
#     r20.append(performance_dct['recall@020'])
#     ndcg10.append(performance_dct['ndcg@010'])
#     ncdg20.append(performance_dct['ndcg@020'])

#     # print(r10)

# plt.plot([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4][::-1], r10[::-1], label='recall@10')
# plt.plot([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4][::-1], r20[::-1], label='recall@20')
# plt.plot([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4][::-1], ndcg10[::-1], label='NDCG@10')
# plt.plot([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4][::-1], ncdg20[::-1], label='NDCG@10')
# plt.xticks([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4][::-1])
# plt.xlabel(str(f"Availability of (all) data used (100%={len(list(set(data.train_df['user_id'].tolist())))} IDs)"))
# plt.legend()
# plt.ylabel("Performance on metric")
# plt.savefig(str(f'{model}data_restricted_{dataset}.png'))