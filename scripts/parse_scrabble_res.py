from scrabble.data_model import *
from scrabble.common import *
from scrabble.eval_func import *
import pdb

#for res in ResultHistory.objects(target_building='ap_m'):
#    print('{0} at {1}'.format(res.source_building_list,
#        res._object_key['pk'].generation_time))
#    history = res.history
#    pdb.set_trace()

target_naes = ['513', '604', '514']

config = {
        'target_building': 'ap_m',
        #'source_building_list': ['bml', 'ebu3b', 'ap_m'],
        'source_building_list': ['ap_m'],
        #'target_building': 'ap_m',
        #'source_building_list': ['ebu3b', 'ap_m'],
        'sample_num_list': [200, 200, 10],
        #'sample_num_list': [200, 200, 10],
        'use_brick_flag': True,
        'negative_flag': True,
        'task': 'scrabble',
        #'crfalgo': 'ap'
        }

#postfixes = ['10', '11']
postfixes = ['0', '2']
#postfixes = ['0', '3']

target_building = config['target_building']
source_buildings = config['source_building_list']

for postfix in postfixes:
    config['postfix'] = postfix
    pdb.set_trace()
    res = query_result(config)
    history = res['history']
    res = []
    for hist in history:
        pred = hist['pred']
        target_srcids = list([srcid for srcid in pred.keys() if srcid[0:3] in target_naes])
        pred = {srcid: hist['pred'][srcid] for srcid in target_srcids}
        truth = get_true_labels(target_srcids, 'tagsets')
        curr_mf1 = get_macro_f1(truth, pred)
        curr_acc = get_accuracy(truth, pred)
        num_learning_srcids = hist['learning_srcids']
        num_learning_srcids -= 200 * (len(config['source_building_list']) -1 )
        res.append({
            'metrics': {
                'accuracy': curr_acc,
                'macrof1-all': curr_mf1,
                },
            'learning_srcids': num_learning_srcids
            })
    with open('result/allentities_transfer_scrabble_{target_building}_{source_buildings}_{exp_id}_brick.json'.format(
        target_building = target_building,
        source_buildings = '_'.join(source_buildings),
        exp_id = postfix,
        ), 'w')  as fp:
        json.dump(res, fp)
