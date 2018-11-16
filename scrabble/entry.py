import arrow
import pdb
import sys
import json
from copy import deepcopy
import os
import random

from .data_model import *
from .common import *
from .char2ir import Char2Ir

argparser = argparse.ArgumentParser()
argparser.register('type','bool',str2bool)
argparser.register('type','slist', str2slist)
argparser.register('type','ilist', str2ilist)
argparser.add_argument('-task',
                       type=str,
                       help='Learning model among ["char2ir", "ir2tagsets", "scrabble"]',
                       choices=['char2ir', 'ir2tagsets', 'scrabble', 'tagsets2entities'],
                       dest='task',
                       required=True
                       )
argparser.add_argument('-bl',
                       type='slist',
                       help='Learning source building name list',
                       dest='source_building_list')
argparser.add_argument('-nl',
                       type='ilist',
                       help='A list of the number of learning sample',
                       dest='sample_num_list')
argparser.add_argument('-t',
                       type=str,
                       help='Target buildling name',
                       dest='target_building')
argparser.add_argument('-ub',
                       type='bool',
                       help='Use Brick when learning',
                       default=False,
                       dest='use_brick_flag')
argparser.add_argument('-ut',
                       type='bool',
                       help='Use Known Tags',
                       default=False,
                       dest='use_known_tags')
argparser.add_argument('-iter',
                       type=int,
                       help='Number of iteration for the given work',
                       dest='iter_num',
                       default=20)
argparser.add_argument('-nj',
                       type=int,
                       help='Number of processes for multiprocessing',
                       dest='n_jobs',
                       default=2)
argparser.add_argument('-inc',
                       type=int,
                       help='Inc num in each strage',
                       dest='inc_num',
                       default=10)
argparser.add_argument('-ct',
                       type=str,
                       help='Tagset classifier type. one of RandomForest, \
                          StructuredCC, MLP.',
                       dest='tagset_classifier_type',
                       default='MLP')
argparser.add_argument('-ts',
                       type='bool',
                       help='Flag to use time series features too',
                       dest='ts_flag',
                       default=False)
argparser.add_argument('-neg',
                       type='bool',
                       help='Negative Samples augmentation',
                       dest='negative_flag',
                       default=True)
argparser.add_argument('-post',
                       type=str,
                       help='postfix of result filename',
                       default='0',
                       dest = 'postfix')
argparser.add_argument('-crfqs',
                       type=str,
                       help='Query strategy for CRF',
                       default='confidence',
                       dest = 'crfqs')
argparser.add_argument('-crfalgo',
                       type=str,
                       help='Algorithm for CRF',
                       default='ap',
                       dest = 'crfalgo')
argparser.add_argument('-entqs',
                       type=str,
                       help='Query strategy for CRF',
                       default='phrase_util',
                       dest = 'entqs')
argparser.add_argument('-g',
                       type='bool',
                       help='Graphize the result or not',
                       dest='graphize',
                       default=False)
argparser.add_argument('-basedir',
                       type=str,
                       help='Base directory for storing results and temporal data (default: running directory)',
                       default='.',
                       dest = 'basedir')
argparser.add_argument('-srcidfile',
                       type=str,
                       help='A file with the list of srcids to use as a training srcids in JSON',
                       default='',
                       dest = 'srcidfile')

args = argparser.parse_args()

t0 = arrow.get()

# Generate directories
paths = [args.basedir + loc for loc in ['/temp', '/model', '/result', '/figs']]
for path in paths:
    if not os.path.exists(path):
        os.makedirs(path)

res_obj = get_result_obj(args, True)

source_buildings = args.source_building_list
target_building = args.target_building
source_sample_num_list = args.sample_num_list
framework_type = args.task

building_sentence_dict, target_srcids, building_label_dict,\
    building_tagsets_dict, known_tags_dict = load_data(target_building,
                                                       source_buildings)
tot_tagsets_dict = {}
for building, tagsets_dict in building_tagsets_dict.items():
    tot_tagsets_dict.update(tagsets_dict )

tot_labels_dict = {}
for building, labels_dict in building_label_dict.items():
    tot_labels_dict.update(labels_dict)

t1 = arrow.get()
print(t1-t0)
config = {
    'use_known_tags': args.use_known_tags,
    'n_jobs': args.n_jobs,
    'tagset_classifier_type': args.tagset_classifier_type,
    'use_brick_flag': args.use_brick_flag,
    'crfqs': args.crfqs,
    'crfalgo': args.crfalgo,
    'entqs': args.entqs,
    'negative_flag': args.negative_flag,
}

srcidfile = args.srcidfile

if not srcidfile: # Normal path
    predefined_learning_srcids = []
    for building, source_sample_num in zip(source_buildings,
                                           source_sample_num_list):
        predefined_learning_srcids += select_random_samples(
            building = building,
            srcids = building_tagsets_dict[building].keys(),
            n=source_sample_num,
            use_cluster_flag = True,
            sentence_dict = building_sentence_dict[building],
            shuffle_flag = False
        )
else: # Not commonly used but only for rigid evaluation.
    if not os.path.isfile(srcidfile):
        raise Exception('{0} is not a file'.format(srcidfile))
    with open(srcidfile, 'r') as fp:
        predefined_learning_srcids = json.load(fp)

if framework_type == 'char2ir':
    from scrabble import Scrabble
    scrabble = Scrabble(target_building,
                        target_srcids,
                        building_label_dict,
                        building_sentence_dict,
                        building_tagsets_dict,
                        source_buildings,
                        source_sample_num_list,
                        known_tags_dict={},
                        config=config,
                        learning_srcids=predefined_learning_srcids
                        )
    framework = scrabble.char2ir
elif framework_type == 'ir2tagsets':
    from scrabble import Scrabble
    scrabble = Scrabble(target_building,
                        target_srcids,
                        building_label_dict,
                        building_sentence_dict,
                        building_tagsets_dict,
                        source_buildings,
                        source_sample_num_list,
                        known_tags_dict,
                        config=config,
                        learning_srcids=predefined_learning_srcids,
                        task='ir2tagsets',
                        )
    framework = scrabble.ir2tagsets
elif framework_type == 'tagsets2entities':
    from scrabble import Scrabble
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
    framework = scrabble.tagsets2entities
    entities_dict = framework.map_tags_tagsets()
    framework.graphize(entities_dict)
    sys.exit(1)
elif framework_type == 'scrabble':
    from scrabble import Scrabble
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
    framework = scrabble

framework.update_model([])
history = []
curr_learning_srcids = []
for i in range(0, args.iter_num + 1):
    t2 = arrow.get()
    if framework_type == 'char2ir':
        pred_tags = framework.predict(target_srcids + framework.learning_srcids)
        pred = None
    elif framework_type == 'ir2tagsets':
        pred = framework.predict(target_srcids + scrabble.learning_srcids)
        pred_tags = None
    elif framework_type == 'scrabble':
        pred = framework.predict(target_srcids + scrabble.learning_srcids)
        pred_tags = None

    tot_crf_acc, learning_crf_acc, \
    tot_crf_f1, tot_crf_mf1,\
    learning_crf_f1, learning_crf_mf1,\
    tot_acc, tot_point_acc,\
    learning_acc, learning_point_acc = calc_acc(
            true      = tot_tagsets_dict,
            pred      = pred,
            true_crf  = tot_labels_dict,
            pred_crf  = pred_tags,
            srcids    = target_srcids,
            learning_srcids = framework.learning_srcids,
            )
    print_status(framework, tot_acc, tot_point_acc,
                 learning_acc, learning_point_acc,
                 tot_crf_acc, learning_crf_acc,
                 tot_crf_f1, tot_crf_mf1,
                 learning_crf_f1, learning_crf_mf1,
                 )
    new_srcids = [srcid for srcid in set(framework.learning_srcids)
                  if srcid not in curr_learning_srcids]
    if framework_type == 'char2ir':
        hist = {
            'acc': tot_crf_acc,
            'new_srcids': new_srcids,
            'learning_srcids': len(list(set(framework.learning_srcids)))
        }
    else:
        hist = {
            'pred': pred,
            'pred_tags': pred_tags,
            'new_srcids': new_srcids,
            'learning_srcids': len(list(set(framework.learning_srcids)))
        }
    hist['crf_f1'] = tot_crf_f1
    hist['crf_mf1'] = tot_crf_mf1
    curr_learning_srcids = list(set(framework.learning_srcids))
    t3 = arrow.get()
    res_obj.history.append(hist)
    res_obj.save()
    new_srcids = framework.select_informative_samples(args.inc_num)
    framework.update_model(new_srcids)
    print('{0}th took {1}'.format(i, t3 - t2))
