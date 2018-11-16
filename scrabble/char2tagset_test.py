import arrow
import pdb
import json
from copy import deepcopy
import os

from scrabble import Scrabble
from data_model import *
from common import *
import random


args =argparser.parse_args()


t0 = arrow.get()

#target_building = 'ap_m'
#source_buildings = ['ap_m']
#source_buildings = ['ebu3b', 'ap_m']
#source_sample_num_list = [200, 10]
#source_sample_num_list = [200, 10]

res_obj = get_result_obj(args)

source_buildings = args.source_building_list
target_building = args.target_building
source_sample_num_list = args.sample_num_list

building_sentence_dict, target_srcids, building_label_dict,\
    building_tagsets_dict, known_tags_dict = load_data(target_building,
                                                       source_buildings)
tot_label_dict = {}
for building, tagsets_dict in building_tagsets_dict.items():
    tot_label_dict.update(tagsets_dict )

t1 = arrow.get()
print(t1-t0)
config = {
    'use_known_tags': args.use_known_tags,
    'n_jobs': args.n_jobs,
    'tagset_classifier_type': args.tagset_classifier_type,
    'use_brick_flag': args.use_brick_flag,
}

learning_srcid_file = 'metadata/test'
for building, source_sample_num in zip(source_buildings,
                                       source_sample_num_list):
    learning_srcid_file += '_{0}_{1}'.format(building, source_sample_num)
learning_srcid_file += '_srcids.json'

if os.path.isfile(learning_srcid_file):
    with open(learning_srcid_file, 'r') as fp:
        predefined_learning_srcids = json.load(fp)
else:
    predefined_learning_srcids = []
    for building, source_sample_num in zip(source_buildings,
                                           source_sample_num_list):
        predefined_learning_srcids += select_random_samples(building,
                                                 building_tagsets_dict[building].keys(),
                                                 source_sample_num,
                                                 True)
    with open(learning_srcid_file, 'w') as fp:
        json.dump(predefined_learning_srcids, fp)

scrabble = Scrabble(target_building,
                    target_srcids,
                    building_label_dict,
                    building_sentence_dict,
                    building_tagsets_dict,
                    source_buildings,
                    source_sample_num_list,
                    known_tags_dict,
                    config=config,
                    learning_srcids=predefined_learning_srcids
                    )

scrabble.update_model([])
history = []
for i in range(0, 20):
    t2 = arrow.get()
    new_srcids = scrabble.select_informative_samples(10)
    scrabble.update_model(new_srcids)
    pred = scrabble.predict(target_srcids + scrabble.learning_srcids)
    pred_tags = scrabble.predict_tags(target_srcids)
    tot_acc, tot_point_acc, learning_acc, learning_point_acc = \
        calc_acc(tot_label_dict, pred, target_srcids, scrabble.learning_srcids)
    print_status(scrabble, tot_acc, tot_point_acc,
                 learning_acc, learning_point_acc)
    hist = {
        'pred': pred,
        'pred_tags': pred_tags,
        'learning_srcids': len(list(set(scrabble.learning_srcids)))
    }
    t3 = arrow.get()
    res_obj.history.append(hist)
    res_obj.save()
    print('{0}th took {1}'.format(i, t3 - t2))
