#!/bin/bash

# instacart
python run_experiment.py --dataset instacart --model tifuknn --batch-size 1000 --num_nearest_neighbors 300 --within_decay_rate 0.9 --group_decay_rate 0.7 --group_count 3 --alpha 0.7
python run_experiment.py --dataset instacart --model tifuknn --batch-size 1000 --num_nearest_neighbors 900 --within_decay_rate 0.9 --group_decay_rate 0.7 --group_count 3 --alpha 0.7
# tafeng
python run_experiment.py --dataset tafeng --model tifuknn --batch-size 1000 --num_nearest_neighbors 900 --within_decay_rate 0.9 --group_decay_rate 0.6 --group_count 3 --alpha 0.2
python run_experiment.py --dataset tafeng --model tifuknn --batch-size 1000 --num_nearest_neighbors 300 --within_decay_rate 0.9 --group_decay_rate 0.6 --group_count 3 --alpha 0.2
# dunnhumby
python run_experiment.py --dataset dunnhumby --model tifuknn --batch-size 1000 --num_nearest_neighbors 900 --within_decay_rate 0.9 --group_decay_rate 0.7 --group_count 7 --alpha 0.7
python run_experiment.py --dataset dunnhumby --model tifuknn --batch-size 1000 --num_nearest_neighbors 300 --within_decay_rate 0.9 --group_decay_rate 0.7 --group_count 7 --alpha 0.7
