| Data | Metric | $\begin{array}{c}\text { (a) } \\\text { Paper results }\end{array}$ | $\begin{array}{c}\text { (b) } \\\text { Our reproduced results }\end{array}$ |
| :---: | :---: | :---: | :---: |
| ValuedShopper | recall@10 | 0.2162 | 0.2173 |
|  | recall@20 | 0.3028 | 0.2946 |
|  | NDCG@10 | 0.2171 | 0.2251 |
|  | NDCG@20 | 0.2589 | 0.2641 |
| Instacart | recall@10 | 0.3952 | 0.398|
|  | recall@20 | 0.4875 | 0.5015 |
|  | NDCG@10 | 0.3825 | 0.3992 |
|  | NDCG@20 | 0.4384 | 0.4541 |
| Dunnhumby | recall@10 | 0.2087 | 0.2136 |
|  | recall@20 | 0.2692 | 0.2849 |
|  | NDCG@10 | 0.1983 | 0.2031 |
|  | NDCG@20 | 0.2302 | 0.2364 |
| TaFeng | recall@10 | 0.1301 | 0.1277 |
|  | recall@20 | 0.1810 |0.1839 |
|  | NDCG@10 | 0.1011 | 0.1049 |
|  | NDCG@20 | 0.1206 | 0.1264 |

python run.py --historical_records_directory ./data/VS_history_order.csv --future_records_directory ./data/VS_future_order.csv  --n_neighbors 300 --within_decay_rate 1 --group_decay_rate 0.6 --alpha 0.7 --group_size 7 --topk 10

python run.py --historical_records_directory ./data/Instacart_history.csv --future_records_directory ./data/Instacart_future.csv  --n_neighbors 900 --within_decay_rate 0.9 --group_decay_rate 0.7 --alpha 0.7 --group_size 3 --topk 10

python run.py --historical_records_directory ./data/Dunnhumby_history.csv --future_records_directory ./data/Dunnhumby_future.csv  --n_neighbors 900 --within_decay_rate 0.9 --group_decay_rate 0.6 --alpha 0.2 --group_size 3 --topk 10

python run.py --historical_records_directory ./data/TaFang_history_NB.csv --future_records_directory ./data/TaFang_future_NB.csv --n_neighbors 300 --within_decay_rate 0.9 --group_decay_rate 0.7 --alpha 0.7 --group_size 7 --topk 10