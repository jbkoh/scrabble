import arrow
import pdb
import json
import pandas as pd
from collections import defaultdict
from copy import deepcopy

from ir2tagsets import Ir2Tagsets
from data_model import *
from common import *

t0 = arrow.get()


target_building = 'ap_m'
#source_buildings = ['ap_m']
#source_sample_num_list = [10]
source_buildings = ['ebu3b', 'ap_m']
source_sample_num_list = [200, 10]
building_sentence_dict, target_srcids, building_label_dict,\
    building_tagsets_dict, known_tags_dict= load_data(target_building, source_buildings)

tot_label_dict = {}
for building, tagsets_dict in building_tagsets_dict.items():
    tot_label_dict.update(tagsets_dict )
"""
building_sentence_dict = dict()
building_label_dict = dict()
building_tagsets_dict = dict()
for building in source_buildings:
    true_tagsets = {}
    label_dict = {}
    for labeled in LabeledMetadata.objects(building=building):
        srcid = labeled.srcid
        true_tagsets[srcid] = labeled.tagsets
        fullparsing = None
        for clm in column_names:
            one_fullparsing = [i[1] for i in labeled.fullparsing[clm]]
            if not fullparsing:
                fullparsing = one_fullparsing
            else:
                fullparsing += ['O'] + one_fullparsing
                #  This format is alinged with the sentence 
                #  configormation rule.
        label_dict[srcid] = fullparsing

    building_tagsets_dict[building] = true_tagsets
    building_label_dict[building] = label_dict
    sentence_dict = dict()
    for raw_point in RawMetadata.objects(building=building):
        srcid = raw_point.srcid
        metadata = raw_point['metadata']
        if srcid in true_tagsets:
            sentence = None
            for clm in column_names:
                if not sentence:
                    sentence = [c for c in metadata[clm].lower()]
                else:
                    sentence += ['\n'] + \
                                [c for c in metadata[clm].lower()]
            sentence_dict[srcid]  = sentence
        known_tags_dict[srcid] += units[metadata.get('BACnetUnit')]
        #known_tags_dict[srcid] += bacnettypes[metadata.get('BACnetTypeStr')]
    building_sentence_dict[building] = sentence_dict
"""

known_tags_dict = dict(known_tags_dict)
target_srcids = list(building_label_dict[target_building].keys())
t1 = arrow.get()
print(t1-t0)
ir2tagsets = Ir2Tagsets(target_building,
                        target_srcids,
                        building_label_dict,
                        building_sentence_dict,
                        building_tagsets_dict,
                        source_buildings,
                        source_sample_num_list,
                        known_tags_dict=known_tags_dict,
                        config={
                            'use_known_tags': True,
                            'n_jobs':30,
                            'vectorizer_type': 'tfidf',
                            'tagset_classifier_type': 'MLP',
                            'use_brick_flag': True,
                            #'query_strategy': 'phrase_util'
                        }
                        )

history = []
ir2tagsets.update_model([])
for i in range(0, 20):
    t2 = arrow.get()
    new_srcids = ir2tagsets.select_informative_samples(10)
    ir2tagsets.update_model(new_srcids)
    pred = ir2tagsets.predict(target_srcids + ir2tagsets.learning_srcids)
    tot_acc, tot_point_acc, learning_acc, learning_point_acc = \
        calc_acc(tot_label_dict, pred, target_srcids,
                 ir2tagsets.learning_srcids)
    print_status(ir2tagsets, tot_acc, tot_point_acc,
                 learning_acc, learning_point_acc)
    hist = {
        'pred': pred,
        'learning_srcids': list(set(deepcopy(ir2tagsets.learning_srcids))),
    }
    history.append(hist)

    t3 = arrow.get()
    print('{0}th took {1}'.format(i, t3 - t2))

    with open('result/tagsets_only_total_{0}.json'.format(target_building), 'w') as fp:
        json.dump(history, fp)
