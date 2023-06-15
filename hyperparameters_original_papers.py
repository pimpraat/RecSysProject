#TODO: Can someone double check these?
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