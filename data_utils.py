import csv

def partition_the_data_validate(data_chunk, key_set, next_k_step):
    filtered_key_set = []
    past_chunk = 0
    future_chunk = 1
    for key in key_set:
        if len(data_chunk[past_chunk][key]) <= 3:
            continue
        if len(data_chunk[future_chunk][key]) < 2 + next_k_step:
            continue
        filtered_key_set.append(key)

    training_key_set = filtered_key_set[0:int(4 / 5 * len(filtered_key_set)*0.9)]
    validation_key_set = filtered_key_set[int(4 / 5 * len(filtered_key_set)*0.9):int(4 / 5 * len(filtered_key_set))]
    print('Number of training instances: ' + str(len(training_key_set)))
    test_key_set = filtered_key_set[int(4 / 5 * len(filtered_key_set)):]
    return training_key_set, validation_key_set, test_key_set


def generate_dictionary_BA(files, attributes_list):
    # path = '../Minnemudac/'
    #files = ['Coborn_history_order.csv','Coborn_future_order.csv']
    #files = ['BA_history_order.csv', 'BA_future_order.csv']
    #attributes_list = ['MATERIAL_NUMBER']
    dictionary_table = {}
    counter_table = {}
    for attr in attributes_list:
        dictionary = {}
        dictionary_table[attr] = dictionary
        counter_table[attr] = 0

    csv.field_size_limit(sys.maxsize)
    for filename in files:
        count = 0
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=str(','), quotechar=str('|'))
            for row in reader:
                if count == 0:
                    count += 1
                    continue
                key = attributes_list[0]
                if row[2] not in dictionary_table[key]:
                    dictionary_table[key][row[2]] = counter_table[key]
                    counter_table[key] = counter_table[key] + 1
                    count += 1

    print(counter_table)

    total = 0
    for key in counter_table.keys():
        total = total + counter_table[key]

    print('# dimensions of final vector: ' + str(total) + ' | '+str(count-1))

    return dictionary_table, total, counter_table


def read_claim2vector_embedding_file_no_vector(files):
    #attributes_list = ['DRG', 'PROVCAT ', 'RVNU_CD', 'DIAG', 'PROC']
    attributes_list = ['MATERIAL_NUMBER']
    # path = '../Minnemudac/'
    print('start dictionary generation...')
    dictionary_table, num_dim, counter_table = generate_dictionary_BA(files, attributes_list)
    print('finish dictionary generation*****')
    usr_attr = 'CUSTOMER_ID'
    ord_attr = 'ORDER_NUMBER'

    #dictionary_table, num_dim, counter_table = GDF.generate_dictionary(attributes_list)

    freq_max = 200
    ## all the follow three ways array. First index is patient, second index is the time step, third is the feature vector
    data_chunk = []
    day_gap_counter = []
    claims_counter = 0
    num_claim = 0
    code_freq_at_first_claim = np.zeros(num_dim+2)


    for file_id in range(len(files)):

        count = 0
        data_chunk.append({})
        filename = files[file_id]
        with open(filename, 'r') as csvfile:
            #gap_within_one_year = np.zeros(365)
            reader = csv.DictReader(csvfile)
            last_pid_date = '*'
            last_pid = '-1'
            last_days = -1
            # 2 more elements in the end for start and end states
            feature_vector = []
            for row in reader:
                cur_pid_date = row[usr_attr] + '_' + row[ord_attr]
                cur_pid = row[usr_attr]
                #cur_days = int(row[ord_attr])

                if cur_pid != last_pid:
                    # start state
                    tmp = [-1]
                    data_chunk[file_id][cur_pid] = []
                    data_chunk[file_id][cur_pid].append(tmp)
                    num_claim = 0
                # else:
                #     if last_days != cur_days and last_days != -1 and file_id != 0 and file_id != 2:
                #
                #         gap_within_one_year[cur_days - last_days] = gap_within_one_year[cur_days - last_days] + 1


                if cur_pid_date not in last_pid_date:
                    if last_pid_date not in '*' and last_pid not in '-1':
                        sorted_feature_vector = np.sort(feature_vector)
                        data_chunk[file_id][last_pid].append(sorted_feature_vector)
                        if len(sorted_feature_vector) > 0:
                            count = count + 1
                        #data_chunk[file_id][last_pid].append(feature_vector)
                    feature_vector = []

                    claims_counter = 0
                if cur_pid != last_pid:
                    # end state
                    if last_pid not in '-1':

                        tmp = [-1]
                        data_chunk[file_id][last_pid].append(tmp)

                key = attributes_list[0]

                within_idx = dictionary_table[key][row[key]]
                previous_idx = 0

                for j in range(attributes_list.index(key)):
                    previous_idx = previous_idx + counter_table[attributes_list[j]]
                idx = within_idx + previous_idx

                # set corresponding dimention to 1
                if idx not in feature_vector:
                    feature_vector.append(idx)

                last_pid_date = cur_pid_date
                last_pid = cur_pid
                #last_days = cur_days
                if file_id == 1:
                    claims_counter = claims_counter + 1


            if last_pid_date not in '*' and last_pid not in '-1':
                data_chunk[file_id][last_pid].append(np.sort(feature_vector))
        # if file_id != 0 and file_id != 2:
        #     day_gap_counter.append(gap_within_one_year)
        # print('num of vectors having entries more than 1: ' + str(count))
  #  print(len(data_chunk[0]))

    #print(data_chunk[0]['33050811449.0'])

#    print(data_chunk[0]['33051194484.0'])

 #   print(data_chunk[0]['33051313687.0'])

    return data_chunk, num_dim + 2, code_freq_at_first_claim