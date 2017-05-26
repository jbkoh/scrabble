import json

filenames = ['biased_result', 'unbiased_result']
config_list = ['building_list', 'sample_num_list', 'target_building', 'token_type', 'label_type', 'use_cluster_flag', 'use_brick_flag', 'debug_flag', 'eda_flag', 'ts_flag', 'nj']

for filename in filenames:
    results = list()
    item_flag = False
    with open(filename, 'r') as fp:
        result = dict()
        for line in fp.readlines():
            if line[0:2] == '([':
                result = dict()
                result['config'] = dict()
                result['result'] = dict()
                line = line.replace('200,', '200;')
                config_values = line[2:-2].split(', ')


                for config_key, config_value in zip(config_list, config_values):
                    result['config'][config_key] = config_value
                item_flag = True
            elif item_flag and line[0:3] == 'Ave':
                result_key = line.split(':')[0]
                result_value = float(line.split(':')[-1])
                result['result'][result_key] = result_value
            else:
                item_flag = False
                if result:
                    results.append(result)
                    result = dict()
    with open(filename + '.json', 'w') as fp:
        json.dump(results, fp, indent=2)
