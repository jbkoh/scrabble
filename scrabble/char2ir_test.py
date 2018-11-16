import arrow
import pdb
from copy import deepcopy
import json

from char2ir import Char2Ir
from data_model import *
from common import *

t0 = arrow.get()

column_names = ['VendorGivenName', 
                 'BACnetName', 
                 'BACnetDescription']


target_building = 'ap_m'
source_buildings = ['ap_m']
source_sample_num_list = [10]


building_sentence_dict = dict()
building_label_dict = dict()
building_tagsets_dict = dict()
for building in source_buildings:
    true_tagsets = {}
    label_dict = {}
    for labeled in LabeledMetadata.objects(building=building):
        srcid = labeled.srcid
        true_tagsets[srcid] = labeled.tagsets
        fullparsing = labeled.fullparsing
        labels = {}
        for metadata_type, pairs in fullparsing.items():
            labels[metadata_type] = [pair[1] for pair in pairs]
        label_dict[srcid] = labels

    building_tagsets_dict[building] = true_tagsets
    building_label_dict[building] = label_dict
    sentence_dict = dict()
    for raw_point in RawMetadata.objects(building=building):
        srcid = raw_point.srcid
        if srcid in true_tagsets:
            metadata = raw_point['metadata']
            sentences = {}
            for clm in column_names:
                if clm in metadata:
                    sentences[clm] = [c for c in metadata[clm].lower()]
            sentence_dict[srcid]  = sentences
    building_sentence_dict[building] = sentence_dict

target_srcids = list(building_label_dict[target_building].keys())
t1 = arrow.get()
print(t1-t0)

char2ir = Char2Ir(target_building, 
                  target_srcids,
                  building_label_dict,
                  building_sentence_dict,
                  source_buildings,
                  source_sample_num_list,
                  conf={
                      'use_cluster_flag': True,
                      #'use_brick_flag': False
                  })

char2ir.update_model([])

history = []
for i in range(0,20):
    t1 = arrow.get()
    new_srcids = char2ir.select_informative_samples(10)
    char2ir.update_model(new_srcids)
    t2 = arrow.get()
    print('{0}th took: {1}'.format(i, t2-t1))
    pred = char2ir.predict(target_srcids)
    metrics = char2ir.evaluate(pred)
    hist = {
        'pred_tags': pred,
        'metrics': metrics,
        'learning_srcids': list(set(deepcopy(char2ir.learning_srcids)))
    }
    history.append(hist)
    with open('result/test_crfonly.json', 'w') as fp:
        json.dump(history, fp)

slack_notifier('CRF only done')
