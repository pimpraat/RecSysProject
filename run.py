from __future__ import unicode_literals, print_function, division
import numpy as np
import sys

import argparse

from src.data_utils import read_claim2vector_embedding_file_no_vector, partition_the_data_validate
from src.tifuknn import KNN, temporal_decay_sum_history, merge_history, predict_with_elements_in_input
from src.metric_utils import get_precision_recall_Fscore, get_HT, get_NDCG1


activate_codes_num = -1
next_k_step = 1
training_chunk = 0
test_chunk = 1


def evaluate(data_chunk,  training_key_set, test_key_set, input_size, group_size,
             within_decay_rate, group_decay_rate, num_nearest_neighbors, alpha,  topk):
    activate_codes_num = -1
    temporal_decay_sum_history_training = temporal_decay_sum_history(data_chunk[training_chunk],
                                                                     training_key_set, input_size,
                                                                     group_size, within_decay_rate,
                                                                     group_decay_rate)
    temporal_decay_sum_history_test = temporal_decay_sum_history(data_chunk[training_chunk],
                                                                 test_key_set, input_size,
                                                                 group_size, within_decay_rate,
                                                                 group_decay_rate)
    index, distance = KNN(temporal_decay_sum_history_test, temporal_decay_sum_history_training,
                          num_nearest_neighbors)


    sum_history = merge_history(temporal_decay_sum_history_test, test_key_set, temporal_decay_sum_history_training,
                                training_key_set, index, alpha)


    if activate_codes_num < 0:
        # for i in range(1, 6):

        prec = []
        rec = []
        F = []
        prec1 = []
        rec1 = []
        F1 = []
        prec2 = []
        rec2 = []
        F2 = []
        prec3 = []
        rec3 = []
        F3 = []
        NDCG = []
        n_hit = 0

        num_ele = topk
        # print('k = ' + str(activate_codes_num))
        # evaluate(data_chunk, input_size,test_KNN_history, test_key_set, next_k_step)
        count = 0
        for iter in range(len(test_key_set)):
            # training_pair = training_pairs[iter - 1]
            # input_variable = training_pair[0]
            # target_variable = training_pair[1]
            input_variable = data_chunk[training_chunk][test_key_set[iter]]
            target_variable = data_chunk[test_chunk][test_key_set[iter]]

            if len(target_variable) < 2 + next_k_step:
                continue
            count += 1
            output_vectors = predict_with_elements_in_input(sum_history, test_key_set[iter])
            top = 400
            hit = 0
            for idx in range(len(output_vectors)):
                # for idx in [2]:

                output = np.zeros(input_size)
                target_topi = output_vectors[idx].argsort()[::-1][:top]
                c = 0
                for i in range(top):
                    if c >= num_ele:
                        break
                    output[target_topi[i]] = 1
                    c += 1

                vectorized_target = np.zeros(input_size)
                for ii in target_variable[1 + idx]:
                    vectorized_target[ii] = 1
                precision, recall, Fscore, correct = get_precision_recall_Fscore \
                    (vectorized_target, output)
                prec.append(precision)
                rec.append(recall)
                F.append(Fscore)
                if idx == 0:
                    prec1.append(precision)
                    rec1.append(recall)
                    F1.append(Fscore)
                elif idx == 1:
                    prec2.append(precision)
                    rec2.append(recall)
                    F2.append(Fscore)
                elif idx == 2:
                    prec3.append(precision)
                    rec3.append(recall)
                    F3.append(Fscore)
                hit += get_HT(vectorized_target, target_topi, num_ele)
                ndcg = get_NDCG1(vectorized_target, target_topi, num_ele)
                NDCG.append(ndcg)
            if hit == next_k_step:
                n_hit += 1


        # print('average precision of ' + ': ' + str(np.mean(prec)) + ' with std: ' + str(np.std(prec)))
        recall = np.mean(rec)
        ndcg = np.mean(NDCG)
        hr = n_hit / len(test_key_set)

    return recall, ndcg, hr


def main(args):

    files = [args.historical_records_directory, args.future_records_directory]

    data_chunk, input_size, code_freq_at_first_claim = read_claim2vector_embedding_file_no_vector(files)
    training_key_set, validation_key_set, test_key_set = partition_the_data_validate(data_chunk, list(data_chunk[test_chunk]), 1)

    print('Num. of top: ', args.topk)
    recall, ndcg, hr = evaluate(data_chunk, training_key_set, test_key_set, input_size,
                                args.group_size, args.within_decay_rate, args.group_decay_rate,
                                args.n_neighbors, args.alpha,  args.topk)

    print('recall: ', str(np.round(recall, 4)))
    print('NDCG: ', str(np.round(ndcg, 4)))
    sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--historical_records_directory', default="./data/TaFang_history_NB.csv", type=str, help='')
    parser.add_argument('--future_records_directory', default="./data/TaFang_future_NB.csv", type=str, help='')
    parser.add_argument('--n_neighbors', default=300, type=int, help='')
    parser.add_argument('--within_decay_rate', default=0.9, type=float, help='')
    parser.add_argument('--group_decay_rate', default=0.7, type=float, help='')
    parser.add_argument('--alpha', default=0.7, type=float, help='')
    parser.add_argument('--group_size', default=7, type=int, help='')
    parser.add_argument('--topk', default=10, type=int, help='')

    args = parser.parse_args()
    main(args)

