import numpy as np
from sklearn.neighbors import NearestNeighbors


next_k_step = 1
test_chunk = 1


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


def add_history(data_history, training_key_set, output_size):
    sum_history = {}
    for key in training_key_set:
        sum_vector = np.zeros(output_size)
        count = 0
        for lst in data_history[key]:
            vec = np.zeros(output_size)
            for ele in lst:
                vec[ele] = 1
            if vec[-2] == 1 or vec[-1] == 1:
                continue
            sum_vector += vec
            count += 1
        sum_vector = sum_vector / count
        sum_history[key] = sum_vector
    return sum_history


def temporal_decay_add_history(data_set, key_set, output_size, within_decay_rate):
    sum_history = {}
    for key in key_set:
        vec_list = data_set[key]
        num_vec = len(vec_list) - 2
        his_list = np.zeros(output_size)
        for idx in range(1,num_vec+1):
            his_vec = np.zeros(output_size)
            decayed_val = np.power(within_decay_rate,num_vec-idx)
            for ele in vec_list[idx]:
                his_vec[ele] = decayed_val
            his_list += his_vec
        sum_history[key] = his_list/num_vec
        # sum_history[key] = np.multiply(his_list / num_vec, IDF)

    return sum_history


def KNN(query_set, target_set, k):
    history_mat = []
    for key in target_set.keys():
        history_mat.append(target_set[key])
    test_mat = []
    for key in query_set.keys():
        test_mat.append(query_set[key])
    # print('Finding k nearest neighbors...')
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(history_mat)
    distances, indices = nbrs.kneighbors(test_mat)
    # print('Finish KNN search.' )
    return indices,distances


def weighted_aggragate_outputs(data_chunk, training_key_set, index, distance, output_size):
    output_vectors = []
    key_set = training_key_set
    for index_list_id in range(len(index)):
        outputs = []
        for vec_idx in range(1, next_k_step+1):

            target_vec_list = []
            weight_list = []
            for id in range(len(index[index_list_id])):
                dis = distance[index_list_id][id]
                if dis == 0:
                    weight_list.append(0)
                else:
                    weight_list.append(1 / dis)
            new_weight = softmax(weight_list)
            for i in range(len(new_weight)):
                if new_weight[i] == 0:
                    new_weight[i] = 1
            vec = np.zeros(output_size)
            for id in range(len(index[index_list_id])):
                idx = index[index_list_id][id]
                target_list = data_chunk[test_chunk][key_set[idx]][vec_idx]
                for ele in target_list:
                    vec[ele] += new_weight[id]
            outputs.append(vec)
        output_vectors.append(outputs)
    return output_vectors


def KNN_history_record1(sum_history, output_size, k):
    history_mat = []
    for key in sum_history.keys():
        history_mat.append(sum_history[key])

    print('Finding k nearest neighbors...')
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(history_mat)
    distances, indices = nbrs.kneighbors(history_mat)
    KNN_history = {}
    key_set = list(sum_history)
    for id in range(len(key_set)):
#    for idx_list in indices:
        idx_list = indices[id]
        NN_history = np.zeros(output_size)
        for idx in idx_list:
            NN_history += sum_history[key_set[idx]]
        NN_history = NN_history / k
        KNN_history[key_set[id]] = NN_history

    return KNN_history


def KNN_history_record2(query_set, sum_history, output_size, k):
    history_mat = []
    for key in sum_history.keys():
        history_mat.append(sum_history[key])
    test_mat = []
    for key in query_set.keys():
        test_mat.append(query_set[key])
    print('Finding k nearest neighbors...')
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(history_mat)
    distances, indices = nbrs.kneighbors(test_mat)
    KNN_history = {}
    key_set = list(query_set)
    training_key_set = list(sum_history)
    for id in range(len(key_set)):
#    for idx_list in indices:
        idx_list = indices[id]
        NN_history = np.zeros(output_size)
        for idx in idx_list:
            NN_history += sum_history[training_key_set[idx]]
        NN_history = NN_history / k
        KNN_history[key_set[id]] = NN_history

    return KNN_history,indices


def group_history_list(his_list, group_size):
    grouped_vec_list = []
    if len(his_list) < group_size:
        #sum = np.zeros(len(his_list[0]))
        for j in range(len(his_list)):
            grouped_vec_list.append(his_list[j])

        return grouped_vec_list, len(his_list)
    else:
        est_num_vec_each_block = len(his_list)/group_size
        base_num_vec_each_block = int(np.floor(len(his_list)/group_size))
        residual = est_num_vec_each_block - base_num_vec_each_block

        num_vec_has_extra_vec = int(np.round(residual * group_size))

        if residual == 0:
            for i in range(group_size):
                if len(his_list)<1:
                    print('len(his_list)<1')
                sum = np.zeros(len(his_list[0]))
                for j in range(base_num_vec_each_block):
                    if i*base_num_vec_each_block+j >= len(his_list):
                        print('i*num_vec_each_block+j')
                    sum += his_list[i*base_num_vec_each_block+j]
                grouped_vec_list.append(sum/base_num_vec_each_block)
        else:

            for i in range(group_size - num_vec_has_extra_vec):
                sum = np.zeros(len(his_list[0]))
                for j in range(base_num_vec_each_block):
                    if i*base_num_vec_each_block+j >= len(his_list):
                        print('i*base_num_vec_each_block+j')
                    sum += his_list[i*base_num_vec_each_block+j]
                    last_idx = i * base_num_vec_each_block + j
                grouped_vec_list.append(sum/base_num_vec_each_block)

            est_num = int(np.ceil(est_num_vec_each_block))
            start_group_idx = group_size - num_vec_has_extra_vec
            if len(his_list) - start_group_idx*base_num_vec_each_block >= est_num_vec_each_block:
                for i in range(start_group_idx,group_size):
                    sum = np.zeros(len(his_list[0]))
                    for j in range(est_num):
                        # if residual+(i-1)*est_num_vec_each_block+j >= len(his_list):
                        #     print('residual+(i-1)*num_vec_each_block+j')
                        #     print('len(his_list)')
                        iidxx = last_idx + 1+(i-start_group_idx)*est_num+j
                        if  iidxx >= len(his_list) or iidxx<0:
                            print('last_idx + 1+(i-start_group_idx)*est_num+j')
                        sum += his_list[iidxx]
                    grouped_vec_list.append(sum/est_num)

        return grouped_vec_list, group_size


def temporal_decay_sum_history(data_set, key_set, output_size, group_size, within_decay_rate, group_decay_rate):
    sum_history = {}
    for key in key_set:
        vec_list = data_set[key]
        num_vec = len(vec_list) - 2
        his_list = []
        for idx in range(1,num_vec+1):
            his_vec = np.zeros(output_size)
            decayed_val = np.power(within_decay_rate,num_vec-idx)
            for ele in vec_list[idx]:
                his_vec[ele] = decayed_val
            his_list.append(his_vec)

        grouped_list,real_group_size = group_history_list(his_list,group_size)
        his_vec = np.zeros(output_size)
        for idx in range(real_group_size):
            decayed_val = np.power(group_decay_rate, group_size - 1 - idx)
            if idx>=len(grouped_list):
                print( 'idx: '+ str(idx))
                print('len(grouped_list): ' + str(len(grouped_list)))
            his_vec += grouped_list[idx]*decayed_val
        sum_history[key] = his_vec/real_group_size
        # sum_history[key] = np.multiply(his_vec / real_group_size, IDF)
    return sum_history


def most_frequent_elements(data_chunk, index, training_key_set, output_size):
    output_vectors = []

    for vec_idx in range(1,next_k_step+1):
        vec = np.zeros(output_size)
        for idx in index:
            target_vec = data_chunk[test_chunk][training_key_set[idx]][vec_idx]
            for ele in target_vec:
                vec[ele] += 1

        output_vectors.append(vec)
    return output_vectors


def predict_with_elements_in_input(sum_history, key):
    output_vectors = []

    for idx in range(next_k_step):
        vec = sum_history[key]
        output_vectors.append(vec)
    return output_vectors


def merge_history(sum_history_test, test_key_set, training_sum_history_test, 
                  training_key_set, index, alpha):
    merged_history = {}
    for test_key_id in range(len(test_key_set)):
        test_key = test_key_set[test_key_id]
        test_history = sum_history_test[test_key]
        sum_training_history = np.zeros(len(test_history))
        for indecis in index[test_key_id]:
            training_key = training_key_set[indecis]
            sum_training_history += training_sum_history_test[training_key]

        sum_training_history = sum_training_history/len(index[test_key_id])

        merge = test_history*alpha + sum_training_history*(1-alpha)
        merged_history[test_key] = merge

    return merged_history


def merge_history_and_neighbors_future(future_data, sum_history_test, test_key_set,training_sum_history_test,
                                       training_key_set,index,alpha, beta):
    merged_history = {}
    for test_key_id in range(len(test_key_set)):
        test_key = test_key_set[test_key_id]
        test_history = sum_history_test[test_key]
        sum_training_history = np.zeros(len(test_history))
        sum_training_future = np.zeros(len(test_history))
        for indecis in index[test_key_id]:
            training_key = training_key_set[indecis]
            sum_training_history += training_sum_history_test[training_key]
            # future_vec = np.zeros((len(test_history)))
            for idx in future_data[training_key][1]:
                if idx >= 0:
                    sum_training_future[idx] += 1

        sum_training_history = sum_training_history/len(index[test_key_id])
        sum_training_future = sum_training_future/len(index[test_key_id])

        merge = (test_history*alpha + sum_training_history*(1-alpha))* beta + sum_training_future*(1-beta)
        merged_history[test_key] = merge

    return merged_history