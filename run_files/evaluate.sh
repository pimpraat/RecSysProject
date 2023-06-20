#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=run_experiments
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --mem=32000M
#SBATCH --output=model_inference_1.out

module purge
module load 2022
module load Anaconda3/2022.05
source activate nbr


# Started run at 11.00
# srun python final_evaluation.py --dataset dunnhumby --model tifuknn
# srun python final_evaluation.py --dataset instacart --model tifuknn
# srun python final_evaluation.py --dataset tafeng --model tifuknn
# srun python final_evaluation.py --dataset taobao --model tifuknn
# srun python final_evaluation.py --dataset tmall --model tifuknn
# srun python final_evaluation.py --dataset valuedshopper --model tifuknn

#Started run at 11.00, finshed both at 11.35
# srun python final_evaluation.py --dataset dunnhumby --model top_personal
# srun python final_evaluation.py --dataset instacart --model top_personal
# srun python final_evaluation.py --dataset tafeng --model top_personal
# srun python final_evaluation.py --dataset taobao --model top_personal
# srun python final_evaluation.py --dataset tmall --model top_personal
# srun python final_evaluation.py --dataset valuedshopper --model top_personal

# Started splitted in two runs at 11.35
srun python final_evaluation.py --dataset dunnhumby --model betavae
srun python final_evaluation.py --dataset instacart --model betavae
# srun python final_evaluation.py --dataset tafeng --model betavae
# srun python final_evaluation.py --dataset valuedshopper --model betavae
# srun python final_evaluation.py --dataset taobao --model betavae
# srun python final_evaluation.py --dataset tmall --model betavae