## TIFUKNN

### Reproducing the results

#### Environment

To reproduce the results you need to create an environment. All required packages are listed in the `requirements.txt` file.

#### Datasets

Before running the code you need to download the datasets. Datasets can be find here:

**Instacart**: https://www.kaggle.com/c/instacart-market-basket-analysis/data \
**Dunnhumby**: https://www.dunnhumby.com/source-files/ \
**Tafeng**: https://www.kaggle.com/chiranjivdas09/ta-feng-grocery-dataset

After downloading the dataset you need to put it in the 'data' directory in the folder with name of the dataset in subfolder 'raw'.


The hierarchy of your repository should look as follows:
```
└── RecSys
    |
    ├── bin
    |
    ├── data
    |   ├── dunnhumby
    |   |   └── raw
    |   |       ├── campaign_desc.csv
    |   |       ├── campaign_table.csv
    |   |       ├── casual_data.csv
    |   |       ├── coupon.csv
    |   |       ├── coupon_redempt.csv
    |   |       ├── hh_demographic.csv
    |   |       ├── product.csv
    |   |       └── transaction_data.csv
    |   |     
    |   ├── instacart
    |   |   └── raw
    |   |       ├── aisles.csv
    |   |       ├── departments.csv
    |   |       ├── order_products__prior.csv
    |   |       ├── order_products__train.csv
    |   |       ├── orders.csv
    |   |       └── products.csv
    |   |
    |   └── tafeng
    |       └── raw
    |           └── ta_feng_all_months_merged.csv
    |
    ├── results
    └── src

```

#### Running the code
To reproduce the results with default hyperparameters you need to run the following command:
```python
python run_experiment.py
```

If you want to change the hyperparameters just add '--argument value' to the command above. 

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
> --verbose (bool, optional): Whether to print progress. Defaults to True. \
> --num_nearest_neighbors (int, optional): Number of nearest neighbors. \
> --within_decay_rate (float, optional): Within decay rate. \
> --group_decay_rate (float, optional): Group decay rate. \
> --group_count (int, optional): Group count. \
> --alpha (float, optional): Alpha.

Results will be saved in the 'results' directory as .txt file with the name indicating arguments used 
`{dataset}_{num_nearest_neighbors}_{within_decay_rate}_{group_decay_rate}_{group_count}_{alpha}.txt`

