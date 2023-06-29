## TIFUKNN

### Reproducing the results

#### Environment

To reproduce the results you need to create an environment.
To create the Conda environment with all the necessary libraries 
```shell
conda env create -f environment.yml
```

To activate the created environment:
```shell
conda activate RecSysProject
```

#### Datasets

Before running the code you need to download the datasets. Datasets can be find here: [Dropbox link](https://www.dropbox.com/scl/fo/9ytigi0278u1zufp8e86a/h?dl=0&rlkey=rj5fbm835r43pfltblpxu1nbi).
This link contain also (preproccesed) data for reproducibility convencience. Unpacking this in the data folder should result in the same directory structure as mentioned below.


The hierarchy of your repository should look as follows:
```
└── RecSys
    |
    ├── run_files
    |
    ├── data
    |   ├── dunnhumby
    |   |   └── raw
    |   |       ├── Dunnhumby_future.csv
    |   |       └── Dunnhumby_history.csv
    |   |     
    |   ├── instacart
    |   |   └── raw
    |   |       ├── Instacart_future.csv
    |   |       └── Instacart_history.csv
    |   |
    |   ├── tafeng
    |   |   └── raw
    |   |       ├── TaFang_future_NB.csv
    |   |       └── TaFang_history_NB.csv
    |   |
    |   ├── taobao
    |   |   └── raw
    |   |       └── taobao.csv
    |   |
    |   ├── tmall
    |   |   └── raw
    |   |       └── ijcai2016_taobao.csv
    |   |
    |   └── valuedshopper
    |       └── raw
    |           ├── VS_future_order.csv
    |           └── VS_history_order.csv
    |
    ├── report_results
    ├── results
    └── src

```

#### Running the code
To reproduce the results with default hyperparameters, on for example the Tafeng dataset using the default TIFUKNN model you need to run the following command:
```python
python final_evaluation.py --model tifuknn --dataset tafeng
```
This automatically saves/updates the results in the 'results' directory in the 'data.json' file.

If you want to change the hyperparameters, or perform parameter tuning just add '--argument value' to the command below. 

For example:
```python
python run_experiment.py --num_nearest_neighbors 100
```

List of all avaliable arguments:
> --dataset (str): Name of the dataset. \
> --model (str): Name of the model. \
> --cutoff_list (list): List of cutoffs. \
> --batch_size (int): Batch size. \
> --dataset_dir_name (str): Name of the dataset directory. \
> --num_nearest_neighbors (int, optional): Number of nearest neighbors. \
> --within_decay_rate (float, optional): Within decay rate. \
> --group_decay_rate (float, optional): Group decay rate. \
> --group_count (int, optional): Group count. \
> --alpha (float, optional): Alpha. \
> --hypertuning (float, optional): Whether to perform hyperparamter tuning. \
> --save_user_metrics (float, optional): Whether to save the metrics per user_id for further analysis.


In this case results will be saved in the 'results' directory as .txt file with the name indicating arguments used 
`{dataset}_{num_nearest_neighbors}_{within_decay_rate}_{group_decay_rate}_{group_count}_{alpha}.txt`

#### Job files
In `run_files` directory you can find job files used for running the experiments on CPU/GPU. 
Commands below will run all experiments used in our study with specified hyperparameters and save the results
to the `results` directory.

CPU:
```shell
bash run_files/run_experiments_cpu.sh
```

GPU (on Lisa):
```shell
sbatch run_files/run_experiments_gpu.job
```

#### Fairness results
To reproduce the fairness results after running the experiments please follow the steps in the `generate_report_results.ipynb` notebook. 