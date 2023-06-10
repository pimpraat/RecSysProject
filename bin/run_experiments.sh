#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=run_experiments
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --mem=32000M
#SBATCH --output=model_inference_%A.out

module purge
module load 2022
module load Anaconda3/2022.05
source activate RecSys

# instacart
srun python -m run_experiments --dataset instacart --model tifuknn --batch-size 1000 --num_nearest_neighbors 300 --within_decay_rate 0.9 --group_decay_rate 0.7 --group_count 3 --alpha 0.7
srun python -m run_experiments --dataset instacart --model tifuknn --batch-size 1000 --num_nearest_neighbors 900 --within_decay_rate 0.9 --group_decay_rate 0.7 --group_count 3 --alpha 0.7
# tafeng
srun python -m run_experiments --dataset tafeng --model tifuknn --batch-size 1000 --num_nearest_neighbors 900 --within_decay_rate 0.9 --group_decay_rate 0.6 --group_count 3 --alpha 0.2
srun python -m run_experiments --dataset tafeng --model tifuknn --batch-size 1000 --num_nearest_neighbors 300 --within_decay_rate 0.9 --group_decay_rate 0.6 --group_count 3 --alpha 0.2
# dunnhumby
srun python -m run_experiments --dataset dunnhumby --model tifuknn --batch-size 1000 --num_nearest_neighbors 900 --within_decay_rate 0.9 --group_decay_rate 0.7 --group_count 7 --alpha 0.7
srun python -m run_experiments --dataset dunnhumby --model tifuknn --batch-size 1000 --num_nearest_neighbors 300 --within_decay_rate 0.9 --group_decay_rate 0.7 --group_count 7 --alpha 0.7
