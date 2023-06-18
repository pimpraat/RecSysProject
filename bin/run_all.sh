#!/bin/bash
# instacart
python run_experiment.py --dataset instacart --model tifuknn --batch-size 1000 --num_nearest_neighbors 900 --within_decay_rate 0.9 --group_decay_rate 0.7 --group_count 3 --alpha 0.9 --save_user_metrics True
# tafeng
python run_experiment.py --dataset tafeng --model tifuknn --batch-size 1000 --num_nearest_neighbors 300 --within_decay_rate 0.9 --group_decay_rate 0.7 --group_count 7 --alpha 0.7 --save_user_metrics True
# dunnhumby
python run_experiment.py --dataset dunnhumby --model tifuknn --batch-size 1000 --num_nearest_neighbors 900 --within_decay_rate 0.9 --group_decay_rate 0.6 --group_count 3 --alpha 0.2 --save_user_metrics True
# valuedshoppers
python run_experiment.py --dataset valuedshopper --model tifuknn --batch-size 1000 --num_nearest_neighbors 300 --within_decay_rate 1 --group_decay_rate 0.6 --group_count 7 --alpha 0.7 --save_user_metrics True
# tmall
python run_experiment.py --dataset tmall --model tifuknn --batch-size 1000 --num_nearest_neighbors 100 --within_decay_rate 0.6 --group_decay_rate 0.8 --group_count 18 --alpha 0.7 --save_user_metrics True
# taobao
python run_experiment.py --dataset taobao --model tifuknn --batch-size 1000 --num_nearest_neighbors 300 --within_decay_rate 0.6 --group_decay_rate 0.8 --group_count 10 --alpha 0.1 --save_user_metrics True
