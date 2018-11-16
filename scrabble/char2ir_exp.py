import arrow
import pdb
import json

from char2ir_gpu import Char2Ir
from data_model import *
from cmd_interface import argparser

args = argparser.parse_args()
print('Configuration: {0}'.format(args))

column_names = ['VendorGivenName',
                 'BACnetName',
                 'BACnetDescription']

# Config phase
if args.target_building:
    target_building = args.target_building
else:
    target_building = 'ebu3b'
if args.buildings:
    source_buildings = args.buildings
else:
    source_buildings = ['ap_m', 'ebu3b']

if args.sample_nums:
    source_sample_num_list = args.sample_nums
else:
    source_sample_num_list = [10, 5]

if args.step_num:
    step_num = args.step_num
else:
    step_num = 10

if args.iter_num:
    iter_num = args.iter_num
else:
    iter_num = 20

# TODO: Print configuration

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
    misc_dict = dict()
    for raw_point in RawMetadata.objects(building=building):
        srcid = raw_point.srcid
        if srcid in true_tagsets:
            metadata = raw_point['metadata']
            sentence = None
            for clm in column_names:
                if not sentence:
                    sentence = [c for c in metadata[clm].lower()]
                else:
                    sentence += ['\n'] + \
                                [c for c in metadata[clm].lower()]
            misc = {
                'unit': metadata['BACnetUnit'],
                'type': metadata['BACnetTypeStr']
            }
            sentence_dict[srcid]  = sentence
            misc_dict[srcid] = misc
    building_sentence_dict[building] = sentence_dict

target_srcids = list(building_label_dict[target_building].keys())

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
#resulter = resulter()

metrics = []
for i in range(0, iter_num):
    t0 = arrow.get()
    new_srcids = char2ir.select_informative_samples(step_num)
    print('pass')
    for srcid in new_srcids:
        if srcid in char2ir.learning_srcids:
            print('WARNING: {0} is selected again.'.format(srcid))
    new_srcids.append('506_2_3010317')
    char2ir.update_model(new_srcids)
    pred = char2ir.predict([srcid for srcid in target_srcids
                            if srcid not in char2ir.learning_srcids])
    res = char2ir.evaluate(pred)
    metrics.append(res)
    t1 = arrow.get()
    print('{0}th took: {1}'.format(i, t1-t0))
    print('Metrics: {0}'.format(res))
#proba = char2ir.predict_proba(target_srcids)
print('result: ')
print(metrics)
with open('result/crfonly_20180428.json', 'w') as fp:
    json.dump(metrics)
