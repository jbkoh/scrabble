import os
from functools import reduce, partial
import json
import random
from collections import OrderedDict, defaultdict, Counter
import pdb
from copy import deepcopy
from operator import itemgetter
from itertools import islice
import argparse
import logging
from imp import reload
from uuid import uuid4
def gen_uuid():
    return str(uuid4())
from multiprocessing import Pool, Manager, Process
import code
import sys
from math import ceil, floor

import pycrfsuite
import pandas as pd
import numpy as np
from IPython.display import Audio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from bson.binary import Binary as BsonBinary
import arrow
from pygame import mixer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import vstack, csr_matrix, hstack, issparse, coo_matrix,\
                         lil_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neural_network import MLPClassifier
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as hier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
#from skmultilearn.ensemble import RakelO, LabelSpacePartitioningClassifier
from skmultilearn.problem_transform import LabelPowerset, ClassifierChain, \
                                           BinaryRelevance
#from skmultilearn.cluster import IGraphLabelCooccurenceClusterer
#from skmultilearn.ensemble import LabelSpacePartitioningClassifier

from resulter import Resulter
from mongo_models import store_model, get_model, get_tags_mapping, \
                         get_crf_results
from entity_recognition import learn_brick_tagsets, \
                               test_brick_tagset, \
                               batch_test_brick_tagset
from brick_parser import pointTagsetList        as  point_tagsets,\
                         locationTagsetList     as  location_tagsets,\
                         equipTagsetList        as  equip_tagsets,\
                         pointSubclassDict      as  point_subclass_dict,\
                         equipSubclassDict      as  equip_subclass_dict,\
                         locationSubclassDict   as  location_subclass_dict,\
                         tagsetTree             as  tagset_tree
#                         tagsetList as tagset_list
tagset_tree['networkadapter'] = list()
subclass_dict = dict()
subclass_dict.update(point_subclass_dict)
subclass_dict.update(equip_subclass_dict)
subclass_dict.update(location_subclass_dict)
subclass_dict['networkadapter'] = list()
tagset_classifier_type = None

from building_tokenizer import nae_dict

point_tagsets += ['unknown', \
                  #'run_command', \
                  #'low_outside_air_temperature_enable_differential_setpoint', \
                  #'co2_differential_setpoint', 
                  #'pump_flow_status', \
                  #'supply_air_temperature_increase_decrease_step_setpoint',\
                  #'average_exhaust_air_static_pressure_setpoint', \
                  #'chilled_water_differential_pressure_load_shed_command', \
                  # AP_M
                  'average_exhaust_air_pressure_setpoint', 
                  'chilled_water_temperature_differential_setpoint',
                  'outside_air_lockout_temperature_differential_setpoint',
                  'vfd_command',
                  'medium_temperature_hot_water_discharge_temperature_load_shed',
                 ] #TODO: How to add these into the tree structure?

tagset_list = point_tagsets + location_tagsets + equip_tagsets
tagset_list.append('networkadapter')

total_srcid_dict = dict()

logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s')

def init_srcids_dict():
    building_list = ['ebu3b', 'ap_m']
    srcids_dict = dict()
    for building in building_list:
        with open('metadata/{0}_char_label_dict.json'\
                  .format(building), 'r') as fp:
            sentence_label_dict = json.load(fp)
        srcids_dict[building] = list(sentence_label_dict.keys())
    return srcids_dict

raw_srcids_dict = init_srcids_dict()


def init():
    init_srcids_dict()


def play_end_alarm():
    mixer.init()
    mixer.music.load('etc/fins_success.wav')
    mixer.music.play()

def adder(x, y):
    return x+y

def splitter(s):
    return s.split('_')

def save_fig(fig, name, dpi=400):
    pp = PdfPages(name)
    pp.savefig(fig, bbox_inches='tight', pad_inches=0, dpi=dpi)
    pp.close()

def calc_base_features(sentence, features={}, building=None):
    sentenceFeatures = list()
    sentence = ['$' if c.isdigit() else c for c in sentence]
    for i, word in enumerate(sentence):
        features = {
            'word.lower='+word.lower(): 1,
            'word.isdigit': float(word.isdigit())
        }
        #TODO: Changed 1.0 -> 1 
        #      Check if it does not degrade the performance.
        if i==0:
            features['BOS'] = 1
        else:
            features['-1:word.lower=' + sentence[i-1].lower()] = 1
        if i in [0,1]:
            features['SECOND'] = 1
        else:
            features['-2:word.lower=' + sentence[i-2].lower()] = 1
        if i<len(sentence)-1:
            features['+1:word.lower='+sentence[i+1].lower()] = 1
        else:
            features['EOS'] = 1
        if re.match("^[a-zA-Z0-9]*$", word):
            features['SPECIAL'] = 1


        sentenceFeatures.append(features)
    return sentenceFeatures


def calc_features(sentence, building=None):
    sentenceFeatures = list()
    sentence = ['$' if c.isdigit() else c for c in sentence]
    for i, word in enumerate(sentence):
        features = {
            'word.lower='+word.lower(): 1.0,
            'word.isdigit': float(word.isdigit())
        }
        if i==0:
            features['BOS'] = 1.0
        else:
            features['-1:word.lower=' + sentence[i-1].lower()] = 1.0
        if i in [0,1]:
            features['SECOND'] = 1.0
        else:
            features['-2:word.lower=' + sentence[i-2].lower()] = 1.0
        if i<len(sentence)-1:
            features['+1:word.lower='+sentence[i+1].lower()] = 1.0
        else:
            features['EOS'] = 1.0
        sentenceFeatures.append(features)
    return sentenceFeatures

def extend_tree(tree, k, d):
    for curr_head, branches in tree.items():
        if k==curr_head:
            branches.append(d)
        for branch in branches:
            extend_tree(branch, k, d)

def augment_tagset_tree(tagsets):
    global tagset_tree
    global subclass_dict
    for tagset in set(tagsets):
        if '-' in tagset:
            classname = tagset.split('-')[0]
            #tagset_tree[classname].append({tagset:[]})
            extend_tree(tagset_tree, classname, {tagset:[]})
            subclass_dict[classname].append(tagset)
            subclass_dict[tagset] = []


def select_random_samples(building, \
                          srcids, \
                          n, \
                          use_cluster_flag,\
                          token_type='justseparate'):
    cluster_filename = 'model/%s_word_clustering_%s.json' % (building, token_type)
    with open(cluster_filename, 'r') as fp:
        cluster_dict = json.load(fp)

    # Learning Sample Selection
    sample_srcids = set()
    length_counter = lambda x: len(x[1])
    ander = lambda x, y: x and y
    if use_cluster_flag:
        sample_cnt = 0
        sorted_cluster_dict = OrderedDict(
            sorted(cluster_dict.items(), key=length_counter, reverse=True))
        while len(sample_srcids) < n:
            for cluster_num, srcid_list in sorted_cluster_dict.items():
                valid_srcid_list = set(srcid_list)\
                        .intersection(set(srcids))\
                        .difference(set(sample_srcids))
                if len(valid_srcid_list) > 0:
                    sample_srcids.add(\
                            random.choice(list(valid_srcid_list)))
                if len(sample_srcids) >= n:
                    break
    else:
#        random_idx_list = random.sample(\
#                            range(0,len(srcids)),n)
#        sample_srcids = [labeled_srcid_list[i] for i in random_idx_list]
        sample_srcids = random.sample(srcids, n)
    return list(sample_srcids)

def over_sampling(X,Y):
    pass


def learn_crf_model(building_list,
                    source_sample_num_list,
                    token_type='justseparate',
                    label_type='label',
                    use_cluster_flag=False,
                    debug_flag=False,
                    use_brick_flag=False):
    """spec = {
            'source_building': building_list[0],
            'target_building': building_list[1] \
                                if len(building_list)>1 \
                                else building_list[0],
            'source_sample_num': source_sample_num_list[0],
            'label_type': label_type,
            'token_type': token_type,
            'use_cluster_flag': use_cluster_flag
            }
            """
    sample_dict = dict()
    assert(isinstance(building_list, list))
    assert(isinstance(source_sample_num_list, list))
    assert(len(building_list)==len(source_sample_num_list))

    # It does training and testing on the same building but different points.

    """
    crf_model_file = 'model/crf_params_char_{0}_{1}_{2}_{3}_{4}.crfsuite'\
            .format(building_list[0], 
                    token_type, 
                    label_type, 
                    source_sample_num_list[0],
                    'clustered' if use_cluster_flag else 'notclustered')
    """
    crf_model_file = 'temp/{0}.crfsuite'.format(gen_uuid())

    log_filename = 'logs/training_{0}_{1}_{2}_{3}_{4}.log'\
            .format(building_list[0], source_sample_num_list[0], token_type, \
                label_type, 'clustered' if use_cluster_flag else 'unclustered')
    logging.basicConfig(filename=log_filename, 
                        level=logging.DEBUG,
                        format='%(asctime)s %(message)s')
    logging.info("Started!!!")


    ### TRAINING ###
    trainer = pycrfsuite.Trainer(verbose=False, algorithm='pa')
    # algorithm: {'lbfgs', 'l2sgd', 'ap', 'pa', 'arow'}
    trainer.set_params({'feature.possible_states': True,
                        'feature.possible_transitions': True})

    data_available_buildings = []
    for building, source_sample_num in zip(building_list, source_sample_num_list):
        with open("metadata/%s_char_sentence_dict_%s.json" % (building, token_type), "r") as fp:
            sentence_dict = json.load(fp)
        sentence_dict = dict((srcid, [char for char in sentence]) for (srcid, sentence) in sentence_dict.items())

        with open('metadata/{0}_char_category_dict.json'.format(building), 'r') as fp:
            char_category_dict = json.load(fp)
        with open('metadata/{0}_char_label_dict.json'.format(building), 'r') as fp:
            char_label_dict = json.load(fp)

        if label_type == 'label':
            label_dict = char_label_dict
        elif label_Type=='category':
            label_dict = char_category_dict

        if building in data_available_buildings:
            with open("model/fe_%s.json"%building, "r") as fp:
                data_feature_dict = json.load(fp)
            with open("model/fe_%s.json"%building, "r") as fp:
                normalized_data_feature_dict = json.load(fp)
            for srcid in sentence_dict.keys():
                if not normalized_data_feature_dict.get(srcid):
                    normalized_data_feature_dict[srcid] = None

        """
        cluster_filename = 'model/%s_word_clustering_%s.json' % (building, token_type)
        if os.path.isfile(cluster_filename):
            with open(cluster_filename, 'r') as fp:
                cluster_dict = json.load(fp)

        # Learning Sample Selection
        sample_srcid_list = set()
        length_counter = lambda x:len(x[1])
        ander = lambda x,y:x and y
        labeled_srcid_list = list(label_dict.keys())
        if use_cluster_flag:
            sample_cnt = 0
            sorted_cluster_dict = OrderedDict(
                    sorted(cluster_dict.items(), key=length_counter, reverse=True))
            while len(sample_srcid_list) < source_sample_num:
                for cluster_num, srcid_list in sorted_cluster_dict.items():
                    valid_srcid_list = set(srcid_list)\
                            .intersection(set(labeled_srcid_list))\
                            .difference(set(sample_srcid_list))
                    if len(valid_srcid_list) > 0:
                        sample_srcid_list.add(\
                                random.choice(list(valid_srcid_list)))
                    if len(sample_srcid_list) >= source_sample_num:
                        break
        else:
            random_idx_list = random.sample(\
                                range(0,len(labeled_srcid_list)),source_sample_num)
            sample_srcid_list = [labeled_srcid_list[i] for i in random_idx_list]
        """
        sample_srcid_list = select_random_samples(building, \
                                                  label_dict.keys(), \
                                                  source_sample_num, \
                                                  use_cluster_flag)

        """
        # Cluster counting (remove later)
        cluster_counter_dict = dict((cluster_id,0)
                                      for cluster_id 
                                      in cluster_dict.keys())

        for srcid in sample_srcid_list:
            for cluster_id, srcid_list in cluster_dict.items():
                if srcid in srcid_list:
                    cluster_counter_dict[cluster_id] += 1
        """


        for srcid in sample_srcid_list:
            sentence = list(map(itemgetter(0), label_dict[srcid]))
            labels = list(map(itemgetter(1), label_dict[srcid]))
            if building in data_available_buildings:
                data_features = normalized_data_feature_dict[srcid]
            else:
                data_features = None
            trainer.append(pycrfsuite.ItemSequence(
                calc_features(sentence, data_features)), labels)

        sample_dict[building] = list(sample_srcid_list)

    # Learn Brick tags

#    if use_brick_flag:
#        with open('metadata/brick_tags_labels.json', 'r') as fp:
#            tag_label_list = json.load(fp)
#        for tag_labels in tag_label_list:
#            char_tags = [' '] + list(map(itemgetter(0), tag_labels)) + [' ']
#            char_labels = ['O'] + list(map(itemgetter(1), tag_labels)) + ['O']
#            trainer.append(pycrfsuite.ItemSequence(
#                calc_features(char_tags, None)), char_labels)


    # Train and store the model file
    trainer.train(crf_model_file)
    with open(crf_model_file, 'rb') as fp:
        model_bin = fp.read()
    model = {
        'source_list': sample_dict,
        'gen_time': arrow.get().datetime, #TODO: change this to 'date'
        'use_cluster_flag': use_cluster_flag,
        'token_type': 'justseparate',
        'label_type': 'label',
        'model_binary': BsonBinary(model_bin),
        'source_building_count': len(building_list)
    }
    store_model(model)
    os.remove(crf_model_file)

    logging.info("Finished!!!")
    play_end_alarm()

def crf_test(building_list,
        source_sample_num_list,
        target_building,
        token_type='justseparate',
        label_type='label',
        use_cluster_flag=False,
        use_brick_flag=False):
    assert len(building_list) == len(source_sample_num_list)


    source_building_name = building_list[0] #TODO: remove this to use the list

    model_query = {'$and':[]}
    model_metadata = {
        'label_type': label_type,
        'token_type': token_type,
        'use_cluster_flag': use_cluster_flag,
        'source_building_count': len(building_list),
    }
    result_metadata = deepcopy(model_metadata)
    result_metadata['source_cnt_list'] = []
    result_metadata['target_building'] = target_building
    for building, source_sample_num in \
            zip(building_list, source_sample_num_list):
        model_query['$and'].append(
            {'source_list.{0}'.format(building): {'$exists': True}})
        model_query['$and'].append({'$where': \
                                    'this.source_list.{0}.length={1}'.\
                                    format(building, source_sample_num)})
        result_metadata['source_cnt_list'].append([building, source_sample_num])
    model_query['$and'].append(model_metadata)
    model_query['$and'].append({'source_building_count':len(building_list)})
    model = get_model(model_query)
    result_metadata['source_list'] = model['source_list']

    crf_model_file = 'temp/{0}.crfsuite'.format(gen_uuid())
    with open(crf_model_file, 'wb') as fp:
        fp.write(model['model_binary'])

    resulter = Resulter(spec=result_metadata)
    log_filename = 'logs/test_{0}_{1}_{2}_{3}_{4}_{5}.log'\
            .format(source_building_name, 
                    target_building,
                    source_sample_num, 
                    token_type, 
                    label_type, \
                    'clustered' if use_cluster_flag else 'unclustered')
    logging.basicConfig(filename=log_filename, 
                        level=logging.DEBUG,
                        format='%(asctime)s %(message)s')
    logging.info("Started!!!")

    data_available_buildings = []

    with open('metadata/{0}_char_label_dict.json'\
                .format(target_building), 'r') as fp:
        target_label_dict = json.load(fp)
    with open('metadata/{0}_char_sentence_dict_{1}.json'\
                .format(target_building, token_type), 'r') as fp:
        char_sentence_dict = json.load(fp)
    with open('metadata/{0}_sentence_dict_{1}.json'\
                .format(target_building, token_type), 'r') as fp:
        word_sentence_dict = json.load(fp)

    sentence_dict = char_sentence_dict
    sentence_dict = dict((srcid, sentence) 
                         for srcid, sentence 
                         in sentence_dict.items() 
                         if target_label_dict.get(srcid))
#    crf_model_file = 'model/crf_params_char_{0}_{1}_{2}_{3}_{4}.crfsuite'\
#                        .format(source_building_name, 
#                                token_type, 
#                                label_type, 
#                                str(source_sample_num),
#                                'clustered' if use_cluster_flag else 'notclustered')

    tagger = pycrfsuite.Tagger()
    tagger.open(crf_model_file)

    predicted_dict = dict()
    score_dict = dict()
    for srcid, sentence in sentence_dict.items():
        predicted = tagger.tag(calc_features(sentence))
        predicted_dict[srcid] = predicted
        score_dict[srcid] = tagger.probability(predicted)

    precisionOfTrainingDataset = 0
    totalWordCount = 0
    error_rate_dict = dict()


    for srcid, sentence_label in target_label_dict.items():
        sentence = sentence_dict[srcid]
        predicted = predicted_dict[srcid]
        printing_pairs = list()
        orig_label_list = list(map(itemgetter(1), sentence_label))
        resulter.add_one_result(srcid, sentence, predicted, orig_label_list)
        for word, predTag, origLabel in \
                zip(sentence, predicted, orig_label_list):
            printing_pair = [word,predTag,origLabel]
            if predTag==origLabel:
                precisionOfTrainingDataset += 1
                printing_pair = ['O'] + printing_pair
            else:
                printing_pair = ['X'] + printing_pair
            totalWordCount += 1
            printing_pairs.append(printing_pair)
        logging.info("=========== {0} ==== {1} ==================="\
                        .format(srcid, score_dict[srcid]))
        error_rate_dict[srcid] = sum([pair[0]=='X' for pair in printing_pairs])\
                                    /float(len(sentence))
        if 'X' in [pair[0] for pair in printing_pairs]:
            for (flag, word, predTag, origLabel) in printing_pairs:
                logging.info('{:5s} {:20s} {:20s} {:20s}'\
                                .format(flag, word, predTag, origLabel))

    result_file = 'result/test_result_{0}_{1}_{2}_{3}_{4}_{5}.json'\
                    .format(source_building_name,
                            target_building,
                            token_type,
                            label_type,
                            source_sample_num,
                            'clustered' if use_cluster_flag else 'unclustered')
    summary_file = 'result/test_summary_{0}_{1}_{2}_{3}_{4}_{5}.json'\
                    .format(source_building_name,
                            target_building,
                            token_type,
                            label_type,
                            source_sample_num,
                            'clustered' if use_cluster_flag else 'unclustered')

    resulter.serialize_result(result_file)
    resulter.summarize_result()
    resulter.serialize_summary(summary_file)
    resulter.store_result_db()

    score_list = list()
    error_rate_list = list()

    logging.info("Finished!!!!!!")

    for srcid, score in OrderedDict(sorted(score_dict.items(),
                                            key=itemgetter(1),
                                            reverse=True))\
                                                    .items():
        if score==0:
            log_score = np.nan
        else:
            log_score = np.log(score)
        score_list.append(log_score)
        error_rate_list.append(error_rate_dict[srcid])

    error_plot_file = 'figs/error_plotting_{0}_{1}_{2}_{3}_{4}_{5}.pdf'\
                    .format(source_building_name, 
                            target_building,
                            token_type, 
                            label_type, 
                            source_sample_num,
                            'clustered' if use_cluster_flag else 'unclustered')
    i_list = [i for i, s in enumerate(score_list) if not np.isnan(s)]
    score_list = [score_list[i] for i in i_list]
    error_rate_list = [error_rate_list[i] for i in i_list]
    trendline = np.polyfit(score_list, error_rate_list, 1)
    p = np.poly1d(trendline)
    plt.scatter(score_list, error_rate_list, alpha=0.3)
    plt.plot(score_list, p(score_list), "r--")
    save_fig(plt.gcf(), error_plot_file)


def sub_dict_by_key_set(d, ks):
    return dict((k,v) for k, v in d.items() if k in ks)
    #return dict([(k,d[k]) for k in ks])

def leave_one_word(s, w):
    if w in s:
        s = s.replace(w, '')
        s = w + '-' + s
    return s

def _bilou_tagset_phraser(sentence, token_labels):
    phrase_labels = list()
    curr_phrase = ''
    for i, (c, label) in enumerate(zip(sentence, token_labels)):
        if label[2:] in ['right_identifier', 'left_identifier']:
            continue
        tag = label[0]
        if tag=='B':
            if curr_phrase:
            # Below is redundant if the other tags handles correctly.       
                phrase_labels.append(curr_phrase)
            curr_phrase = label[2:]
        elif tag == 'I':
            if curr_phrase != label[2:]:
                phrase_labels.append(curr_phrase)
                curr_phrase = label[2:]
        elif tag == 'L':
            if curr_phrase != label[2:]:
                # Add if the previous label is different                    
                phrase_labels.append(curr_phrase)
            # Add current label                                             
            phrase_labels.append(label[2:])
            curr_phrase = ''
        elif tag == 'O':
            # Do nothing other than pushing the previous label
            if curr_phrase:
                phrase_labels.append(curr_phrase)
            curr_phrase = ''
        elif tag == 'U':
            if curr_phrase:
                phrase_labels.append(curr_phrase)
            phrase_labels.append(label[2:])
        else:
            print('Tag is incorrect in: {0}.'.format(label))
            try:
                assert False
            except:
                pdb.set_trace()
    if curr_phrase != '':
        phrase_labels.append(curr_phrase)
    phrase_labels = [leave_one_word(\
                         leave_one_word(phrase_label, 'left_identifier'),\
                            'right_identifier')\
                        for phrase_label in phrase_labels]
    phrase_labels = list(reduce(adder, map(splitter, phrase_labels), []))
    return phrase_labels


def find_key(tv, d, crit):
    for k, v in d.items():
        if crit(tv, v):
            return k
    return None

def find_keys(tv, d, crit):
    keys = list()
    for k, v in d.items():
        if crit(tv, v):
            keys.append(k)
    return keys

def check_in(x,y):
    return x in y

def build_prefixer(building_name):
    return partial(adder, building_name+'#')

def make_phrase_dict(sentence_dict, token_label_dict, srcid_dict, \
                     eda_flag=False):
    #phrase_dict = OrderedDict()
    phrase_dict = dict()
    for srcid, sentence in sentence_dict.items():
        token_labels = token_label_dict[srcid]
        phrases = _bilou_tagset_phraser(sentence, token_labels)
        remove_indices = list()
        for i, phrase in enumerate(phrases):
            #TODO: Below is heuristic. Is it allowable?
            #if phrase.split('-')[0] in ['building', 'networkadapter',\
            #                            'leftidentifier', 'rightidentifier']:
            if phrase.split('-')[0] in ['leftidentifier', 'rightidentifier']:
                remove_indices.append(i)
                pass
        phrases = [phrase for i, phrase in enumerate(phrases)\
                   if i not in remove_indices]
        """
        if eda_flag:
    #        phrases += phrases
            building_name = find_key(srcid, srcid_dict, check_in)
            assert building_name
            prefixer = build_prefixer(building_name)
            phrases = phrases + list(map(prefixer, phrases))
        """
        #phrase_dict[srcid] = phrases + phrases # TODO: Why did I put this before?
        phrase_dict[srcid] = phrases
    return phrase_dict

def get_multi_buildings_data(building_list, srcids=[], \
                             eda_flag=False, token_type='justseparate'):
    sentence_dict = dict()
    token_label_dict = dict()
    truths_dict = dict()
    sample_srcid_list_dict = dict()
    phrase_dict = dict()
    found_points = list()
    for building in building_list:
        temp_sentence_dict,\
        temp_token_label_dict,\
        temp_phrase_dict,\
        temp_truths_dict,\
        temp_srcid_dict = get_building_data(building, srcids,\
                                                    eda_flag, token_type)
        sentence_dict.update(temp_sentence_dict)
        token_label_dict.update(temp_token_label_dict)
        truths_dict.update(temp_truths_dict)
        phrase_dict.update(temp_phrase_dict)

    assert set(srcids) == set(phrase_dict.keys())
    return sentence_dict, token_label_dict, truths_dict, phrase_dict


def get_building_data(building, srcids, eda_flag=False, \
                      token_type='justseparate'):
    with open('metadata/{0}_char_sentence_dict_{1}.json'\
              .format(building, token_type), 'r') as fp:
        sentence_dict = json.load(fp)
    with open('metadata/{0}_char_label_dict.json'\
              .format(building), 'r') as fp:
        sentence_label_dict = json.load(fp)
    with open('metadata/{0}_ground_truth.json'\
              .format(building), 'r') as fp:
        truths_dict = json.load(fp)
    srcid_dict = {building: srcids}
    sentence_dict = sub_dict_by_key_set(sentence_dict, srcids)
    token_label_dict = dict((srcid, list(map(itemgetter(1), labels))) \
                            for srcid, labels in sentence_label_dict.items())
    token_label_dict = sub_dict_by_key_set(token_label_dict, \
                                                srcids)
    # Remove truths_dict subdictionary if needed
    truths_dict = sub_dict_by_key_set(truths_dict, \
                                                srcids)

    phrase_dict = make_phrase_dict(sentence_dict, token_label_dict, \
                                   srcid_dict, eda_flag)

    return sentence_dict, token_label_dict, phrase_dict,\
            truths_dict, srcid_dict

def lengther(x):
    return len(x)

def value_lengther(x): 
    return len(x[1])

def hier_clustering(d, threshold=3):
    srcids = d.keys()
    tokenizer = lambda x: x.split()
    vectorizer = TfidfVectorizer(tokenizer=tokenizer)
    assert isinstance(d, dict)
    assert isinstance(list(d.values())[0], list)
    assert isinstance(list(d.values())[0][0], str)
    doc = [' '.join(d[srcid]) for srcid in srcids]
    vect = vectorizer.fit_transform(doc)
    #TODO: Make vect aligned to the required format
    z = linkage(vect.toarray(), metric='cityblock', method='complete')
    dists = list(set(z[:,2]))
#    threshold = 3
    #threshold = (dists[2] + dists[3]) / 2
    b = hier.fcluster(z, threshold, criterion='distance')
    cluster_dict = defaultdict(list)
    for srcid, cluster_id in zip(srcids, b):
        cluster_dict[str(cluster_id)].append(srcid)
    return OrderedDict(\
               sorted(cluster_dict.items(), key=value_lengther, reverse=True))

def tagsets_prediction(classifier, vectorizer, binerizer, \
                           phrase_dict, srcids, source_target_buildings, \
                       eda_flag, point_classifier):
    global point_tagsets
    logging.info('Start prediction')
    doc = [' '.join(phrase_dict[srcid]) for srcid in srcids]
    if eda_flag:
        vect_doc = eda_vectorizer(vectorizer, doc, \
                                       source_target_buildings, srcids)
    else:
        vect_doc = vectorizer.transform(doc)

    certainty_dict = dict()
    tagsets_dict = dict()
    logging.info('Start Tagset prediction')
    pred_mat = classifier.predict(vect_doc)
    logging.info('Finished Tagset prediction')
    if not isinstance(pred_mat, np.ndarray):
        try:
            pred_mat = pred_mat.toarray()
        except:
            pred_mat = np.asarray(pred_mat)
    logging.info('Start point prediction')
    point_mat = point_classifier.predict(vect_doc)
    logging.info('Finished point prediction')
    #prob_mat = classifier.predict_proba(vect_doc)
    pred_tagsets_dict = dict()
    pred_certainty_dict = dict()
    pred_point_dict = dict()
    for i, (srcid, pred, point_pred) \
            in enumerate(zip(srcids, pred_mat, point_mat)):
        pred_tagsets_dict[srcid] = binerizer.inverse_transform(\
                                        np.asarray([pred]))[0]
        #pred_tagsets_dict[srcid] = list(binerizer.inverse_transform(pred)[0])
        pred_point_dict[srcid] = point_tagsets[point_pred]
        #pred_vec = [prob[i][0] for prob in prob_mat]
        #pred_certainty_dict[srcid] = pred_vec
        pred_certainty_dict[srcid] = 0
    pred_certainty_dict = OrderedDict(sorted(pred_certainty_dict.items(), \
                                             key=itemgetter(1), reverse=True))
    pdb.set_trace()
    logging.info('Finished prediction')
    return pred_tagsets_dict, pred_certainty_dict, pred_point_dict

def tagsets_evaluation(truths_dict, pred_tagsets_dict, pred_certainty_dict,\
                       srcids, pred_point_dict, phrase_dict, debug_flag=False):
    result_dict = defaultdict(dict)
    sorted_result_dict = OrderedDict()
    incorrect_tagsets_dict = dict()
    correct_cnt = 0
    incorrect_cnt = 0
    point_correct_cnt = 0
    point_incorrect_cnt = 0
    empty_point_cnt = 0
    unknown_reason_cnt = 0
    undiscovered_point_cnt = 0
#    for srcid, pred_tagsets in pred_tagsets_dict.items():
    for srcid in srcids:
        pred_tagsets = pred_tagsets_dict[srcid]
        true_tagsets = truths_dict[srcid]
        one_result = {
            'tagsets': pred_tagsets,
            'certainty': pred_certainty_dict[srcid]
        }
        if set(true_tagsets) == set(pred_tagsets):
            correct_cnt += 1
            one_result['correct?'] = True
            result_dict['correct'][srcid] = one_result
            #result_dict['correct'][srcid] = pred_tagsets
            point_correct_cnt += 1
        else:
            incorrect_cnt += 1
            one_result['correct?'] = False
            result_dict['incorrect'][srcid] = one_result
            true_point = None
            for tagset in true_tagsets:
                if tagset in point_tagsets:
                    true_point = tagset
                    break
            try:
                assert true_point
                found_point = None
                #found_point = pred_point_dict[srcid]
                for tagset in pred_tagsets:
                    if tagset in point_tagsets:
                        found_point = tagset
                        break
                if found_point in ['none', None]:
                    empty_point_cnt += 1
                    assert False
                elif found_point != true_point:
                    point_incorrect_cnt += 1
                    print("INCORRECT POINT ({0}) FOUND: {1} -> {1}"\
                          .format(srcid, true_point, found_point))
                    assert False
                else:
                    unknown_reason_cnt += 1
                    point_correct_cnt += 1
                #    pdb.set_trace()
            except:
                print('point not found')
                unknown_reason_cnt += 1
                if debug_flag:
                    print('####################################################')
                    print('TRUE: {0}'.format(true_tagsets))
                    print('PRED: {0}'.format(pred_tagsets))
                    if true_point:
                        #print('point num in source building: {0}'\
                        #      .format(found_point_cnt_dict[true_point]))
                        pass
                    else:
                        print('no point is included here')
                    source_srcid = None
                    source_idx = None
                    for temp_srcid, tagsets in truths_dict.items():
                        if true_point and true_point in tagsets:
                            source_srcid = temp_srcid
                            source_idx = srcids.index(source_srcid)
                            break
                    print('####################################################')
                    pdb.set_trace()
        sorted_result_dict[srcid] = one_result

    precision_total = point_correct_cnt + point_incorrect_cnt
    recall_total = point_correct_cnt + empty_point_cnt
    if precision_total == 0:
        point_precision = 0
    else:
        point_precision = float(point_correct_cnt) / precision_total
    if recall_total == 0:
        point_recall = 0
    else:
        point_recall = float(point_correct_cnt) / recall_total
    precision = float(correct_cnt) / len(srcids)
    print('------------------------------------result---------------')
    print('point precision: {0}'.format(point_precision))
    print('point recall: {0}'.format(point_recall))
    if empty_point_cnt > 0:
        print('rate points not found in source \
              among sensors where point is not found: \n\t{0}'\
              .format(undiscovered_point_cnt / float(empty_point_cnt)))
    if incorrect_cnt == 0:
        notfoundratio = 0
        incorrectratio = 0
        unknownratio = 0
    else:
        notfoundratio = empty_point_cnt / float(incorrect_cnt)
        incorrectratio = point_incorrect_cnt / float(incorrect_cnt)
        unknownratio = unknown_reason_cnt / float(incorrect_cnt)
    print('sensors where a point is not found: ', notfoundratio, 
                                empty_point_cnt)
    print('sensors where incorrect points are found: ', incorrectratio,
                                      point_incorrect_cnt)
    print('unknown reason: ', unknownratio,
                              unknown_reason_cnt)
    print('-----------')
    result_dict['point_precision'] = point_precision
    result_dict['precision'] = precision
    result_dict['point_recall'] = point_recall
    result_dict['point_correct_cnt'] = point_correct_cnt
    result_dict['point_incorrect_cnt'] = point_incorrect_cnt
    result_dict['unfound_point_cnt'] = empty_point_cnt
    return result_dict

def tagsets_evaluation_deprecated(truths_dict, pred_tagsets_dict, pred_certainty_dict,\
                       srcids, pred_point_dict):
    result_dict = defaultdict(dict)
    sorted_result_dict = OrderedDict()
    incorrect_tagsets_dict = dict()
    correct_cnt = 0
    incorrect_cnt = 0
    point_correct_cnt = 0
    point_incorrect_cnt = 0
    empty_point_cnt = 0
    unknown_reason_cnt = 0
    undiscovered_point_cnt = 0
#    for srcid, pred_tagsets in pred_tagsets_dict.items():
    for srcid in srcids:
        pred_tagsets = pred_tagsets_dict[srcid]
        true_tagsets = truths_dict[srcid]
        one_result = {
            'tagsets': pred_tagsets,
            'certainty': pred_certainty_dict[srcid]
        }
        if set(true_tagsets) == set(pred_tagsets):
            correct_cnt += 1
            one_result['correct?'] = True
            result_dict['correct'][srcid] = one_result
            #result_dict['correct'][srcid] = pred_tagsets
            point_correct_cnt += 1
        else:
            incorrect_cnt += 1
            one_result['correct?'] = False
            result_dict['incorrect'][srcid] = one_result
            true_point = None
            for tagset in true_tagsets:
                if tagset in point_tagsets:
                    true_point = tagset
                    break
            try:
                assert true_point
                found_point = None
                for tagset in pred_tagsets:
                    if tagset in point_tagsets:
                        found_point = tagset
                        break
                if not found_point:
                    empty_point_cnt += 1
                elif found_point != true_point:
                    point_incorrect_cnt += 1
                    print("INCORRECT POINT FOUND: {0} -> {1}"\
                          .format(true_point, found_point))
                else:
                    unknown_reason_cnt += 1
                    point_correct_cnt += 1
            except:
                print('point not found')
                pdb.set_trace()
                unknown_reason_cnt += 1
            if False:
                print('####################################################')
                print('TRUE: {0}'.format(true_tagsets))
                print('PRED: {0}'.format(pred_tagsets))
                if true_point:
                    print('point num in source building: {0}'\
                          .format(found_point_cnt_dict[true_point]))
                else:
                    print('no point is included here')
                source_srcid = None
                source_idx = None
                for temp_srcid, tagsets in learning_truths_dict.items():
                    if true_point and true_point in tagsets:
                        source_srcid = temp_srcid
                        source_idx = learning_srcids.index(source_srcid)
                        source_doc = learning_doc[source_idx]
                        source_vect = learning_vect_doc[source_idx]
                        break
                test_idx = test_srcids.index(srcid)
                target_doc = test_doc[test_idx]
                target_vect = test_vect_doc[test_idx]
                print('####################################################')
                if not found_point and true_point in found_points\
                   and true_point not in ['unknown',\
                        'effective_heating_temperature_setpoint',\
                        'effective_cooling_temperature_setpoint',\
                        'supply_air_flow_setpoint',\
                        'output_frequency_sensor']:

                    pdb.set_trace()
                    pass
        sorted_result_dict[srcid] = one_result

    point_precision = float(point_correct_cnt) \
                        / (point_correct_cnt + point_incorrect_cnt)
    point_recall = float(point_correct_cnt) \
                        / (point_correct_cnt + empty_point_cnt)
    precision = float(correct_cnt) / len(srcids)
    print('------------------------------------result---------------')
    print('point precision: {0}'.format(point_precision))
    print('point recall: {0}'.format(point_recall))
    if empty_point_cnt > 0:
        print('rate points not found in source \
              among sensors where point is not found: \n\t{0}'\
              .format(undiscovered_point_cnt / float(empty_point_cnt)))
    print('sensors where a point is not found: ', empty_point_cnt\
                                               /float(incorrect_cnt),\
                                empty_point_cnt)
    print('sensors where incorrect points are found: ', point_incorrect_cnt\
                                                     /float(incorrect_cnt),\
                                      point_incorrect_cnt)
    print('unknown reason: ', unknown_reason_cnt\
                              /float(incorrect_cnt),\
                              unknown_reason_cnt)
    print('-----------')
    result_dict['point_precision'] = point_precision
    result_dict['precision'] = precision
    result_dict['point_recall'] = point_recall
    result_dict['point_correct_cnt'] = point_correct_cnt
    result_dict['point_incorrect_cnt'] = point_incorrect_cnt
    result_dict['unfound_point_cnt'] = empty_point_cnt
    return result_dict

class custom_multi_label():
    def __init__(self, base_classifier):
        self.base_classifier = base_classifier
        self.base_classifiers = []

    def fit(self, X, Y):
        #X = np.asarray(X)
        #Y = np.asarray(Y)
        class_num = Y.shape[1]
#        self.base_classifiers = [deepcopy(self.base_classifier) \
#                                 for i in range(0, class_num)]
        i = 0
        #for y, classifier in zip(Y.T, self.base_classifiers):
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        for y in Y.T:
            self.base_classifiers.append(deepcopy(self.base_classifier.fit(X, y)))
#            classifier.fit(X, y)
            i += 1
            if i%100==0:
                print(i)

    def predict(self, X):
        return np.asarray([classifier.predict(X.toarray()) \
                           for classifier in self.base_classifiers]).T

class custom_project_multi_label():
    def __init__(self, base_classifier, proj_vectors):
        self.base_classifier = base_classifier
        self.proj_vectors = proj_vectors

    def _proj_x(self, X, proj_vector):
        return np.hstack([X[:, j] for j, v in enumerate(proj_vector) \
                          if v == 1])

    def _X_formatting(self, X):
        if issparse(X):
            return X.todense()
        elif not isinstance(X, np.matrix):
            return np.asmatrix(X)
        else:
            return X

    def fit(self, X, Y):
        X = self._X_formatting(X)
        self.base_classifiers = list()
        for y, proj_vector in zip(Y.T, self.proj_vectors):
            #proj_X = np.hstack([X[:,i] for i, v in enumerate(proj_vector) if v==1])
            proj_X = self._proj_x(X, proj_vector)
            self.base_classifiers.append(deepcopy(self.base_classifier)\
                                        .fit(proj_X, y))
    def predict(self, X):
        Ys = list()
        X = self._X_formatting(X)
        for proj_vector, base_classifier \
                in zip(self.proj_vectors, self.base_classifiers):
#            proj_X = np.hstack([X[:, i] for i, v in enumerate(proj_vector) if v==1])
            proj_X = self._proj_x(X, proj_vector)
            Ys.append(base_classifier.predict(proj_X))
        return np.asarray(Ys).T

def eda_vectorizer(vectorizer, doc, source_target_buildings, srcids):
    global total_srcid_dict
    raw_vect_doc = vectorizer.transform(doc)
    vect_doc = raw_vect_doc.toarray()
    for building in source_target_buildings:
        building_mask = np.array([1 if len(srcid) == 36 \
                            or find_key(srcid, total_srcid_dict, check_in)\
                                == building\
                                    else 0 for srcid in srcids])
        vect_doc = np.hstack([vect_doc] \
                             + [np.asmatrix(building_mask \
                                * vect.toarray()[0]).T \
                            for vect in raw_vect_doc.T])
    return vect_doc

def augment_true_mat_with_superclasses_deprecated_and_incorrect(binerizer, given_label_vect,\
                                       subclass_dict=subclass_dict):
    tagsets = binerizer.inverse_transform([given_label_vect])
    updated_tagsets = [find_keys(tagset, subclass_dict, check_in)\
                       for tagset in tagsets]
    pdb.set_trace()
    return binerizer.transform([list(set(tagsets + updated_tagsets))])

class ProjectClassifier():

    def __init__(self, base_classifier, mask):
        self.base_classifier = base_classifier
        self.base_mask = mask
        """
        for i, tagset in enumerate(self.binerizer.classes_):
            tags = tagset.split('_')
            mask = np.zeros(len(vectorizer.vocabulary))
#            mask = [tag for tag in vectorizer.vocabulary.values()
            for vocab, j in vectorizer.vocabulary.items():
                mask[j] = 1 if vocab in tags else 0
            self.mask_dict[i] = mask
        """


    def fit(self, X, y):
        extended_feature_dims = [i for i in range(0,X.shape[1]) \
                                 if i >= len(self.base_mask)]
        self.mask = np.concatenate([self.base_mask, extended_feature_dims])
        X = X[:, np.where(self.mask==1)[0]]
        self.base_classifier.fit(X, y)

    def predict(self, X):
        assert len(self.mask) == X.shape[1]
        X = X[:, np.where(self.mask==1)[0]]
        return self.base_classifier.predict(X)




class StructuredClassifierChain():

    def __init__(self, base_classifier, binerizer, subclass_dict,
                 vocabulary_dict, n_jobs=1):
        self.n_jobs = n_jobs
        self.vocabulary_dict = vocabulary_dict
        self.subclass_dict = subclass_dict
        self.base_classifier = base_classifier
        self.binerizer = binerizer
        self.upper_y_index_list = list()
        self.lower_y_index_list = list()
        self.base_classifiers = list()
        for i, tagset in enumerate(self.binerizer.classes_):
            found_upper_tagsets = find_keys(tagset, self.subclass_dict, check_in)
            upper_tagsets = [ts for ts in self.binerizer.classes_ \
                             if ts in found_upper_tagsets]
            try:
                assert len(found_upper_tagsets) == len(upper_tagsets)
            except:
                pdb.set_trace()
            self.upper_y_index_list.append([np.where(self.binerizer.classes_ == ts)[0][0]\
                                            for ts in upper_tagsets])
            lower_y_indices = list()
            subclasses = self.subclass_dict.get(tagset)
            if not subclasses:
                subclasses = []
            for ts in subclasses:
                indices = np.where(self.binerizer.classes_ == ts)[0]
                if len(indices)>1:
                    assert False
                elif len(indices==1):
                    lower_y_indices.append(indices[0])
            self.lower_y_index_list.append(lower_y_indices)
            self.base_classifiers.append(deepcopy(self.base_classifier))

    def _augment_X(self, X, Y):
        return np.hstack([X, Y*2])

    def fit(self, X,Y):
        if self.n_jobs == 1:
            return self.serial_fit(X, Y)
        else:
            return self.parallel_fit(X, Y)

    def serial_fit(self, X, Y):
        logging.info('Start fitting')
        X = self.conv_array(X)
        Y = self._augment_labels_superclasses(Y)
        for i, y in enumerate(Y.T):
            sub_Y = Y[:, self.upper_y_index_list[i]]
            augmented_X = self._augment_X(X, sub_Y)
            self.base_classifiers[i].fit(augmented_X, y)
        logging.info('Finished fitting')

    def parallel_fit(self, X,Y):
        p = Pool(self.n_jobs)
        Y = self._augment_labels_superclasses(Y)
        mapped_sub_fit = partial(self.sub_fit, X, Y)
        self.base_classifiers = p.map(mapped_sub_fit, range(0,Y.shape[1]))

    def sub_fit(self, X, Y, i):
        if i%10==0:
            logging.info('{0}th learning step'.format(i))
        base_classifier = deepcopy(self.base_classifier)
        y = Y.T[i]
        sub_Y = Y[:, self.upper_y_index_list[i]]
        augmented_X = self._augment_X(X, sub_Y)
        base_classifier.fit(augmented_X, y)
        return base_classifier

    def sub_fit_proj(self, X, Y, i):
        base_base_classifier = deepcopy(self.base_classifier)
        y = Y.T[i]
        tags = self.binerizer.classes_[i].split('_')
        mask = np.zeros(len(self.vocabulary_dict))
        for vocab, j in self.vocabulary_dict.items():
            mask[j] = 1 if vocab in tags else 0
        base_classifier = ProjectClassifier(base_base_classifier, mask)
        sub_Y = Y[:, self.upper_y_index_list[i]]
        augmented_X = self._augment_X(X, sub_Y)
        base_classifier.fit(augmented_X, y)
        return base_classifier

    def parallel_fit_proj(self, X, Y):
        p = Pool(self.n_jobs)
        Y = self._augment_labels_superclasses(Y)
        mapped_sub_fit = partial(self.sub_fit_proj, X, Y)
        self.base_classifiers = p.map(mapped_sub_fit, range(0,Y.shape[1]))

    def serial_fit_proj(self, X, Y):
        self.base_classifiers = list()
        Y = self._augment_labels_superclasses(Y)
        for i in range(0, Y.shape[1]):
            if i%50 == 0:
                logging.info('{0}th learning step'.format(i))
            self.base_classifiers.append(self.sub_fit_proj(X, Y, i))

    def predict(self, X):
        logging.info('Start predicting')
        X = self.conv_array(X)
        Y = np.zeros((X.shape[0], len(self.binerizer.classes_)))
        for i, (upper_y_indices, base_classifier) \
                in enumerate(zip(self.upper_y_index_list,
                                 self.base_classifiers)):
            try:
                assert sum([i <= y_index for y_index in upper_y_indices]) == 0
            except:
                pdb.set_trace()
            sub_Y = Y[:, upper_y_indices]
            augmented_X = self._augment_X(X, sub_Y)
            pred_y = base_classifier.predict(augmented_X)
            Y[:, i] = pred_y
        Y = self._distill_Y(Y)
        logging.info('Finished predicting')
        return Y

    def _distill_Y(self, Y):
        logging.info('Start distilling')
        new_Y = deepcopy(Y)
        for i, y in enumerate(Y):
            new_y = np.zeros(len(y))
            for j, one_y in enumerate(y):
                subclass_y_indices = self.lower_y_index_list[j]
                if 1 in y[subclass_y_indices]:
                    new_y[j] = 0
                else:
                    new_y[j] = one_y
            new_Y[i] = new_y
        logging.info('Finished distilling')
        return new_Y

    def conv_array(self, d):
        if isinstance(d, np.ndarray):
            return d
        if isinstance(d, np.matrix):
            return np.asarray(d)
        else:
            return d.toarray()

    def conv_matrix(self, d):
        if isinstance(d, np.matrix):
            return d
        elif isinstance(d, np.ndarray):
            return np.matrix(d)
        else:
            return d.todense()

    def _augment_labels_superclasses(self, Y):
        logging.info('Start augmenting label mat with superclasses')
        Y = lil_matrix(Y)
        for i, vect in enumerate(Y):
            tagsets = self.binerizer.inverse_transform(vect)[0]
            updated_tagsets = reduce(adder, [
                                find_keys(tagset, self.subclass_dict, check_in)
                                for tagset in tagsets], [])
            #TODO: This is bad code. need to be fixed later.
            finished = False
            while not finished:
                try:
                    new_row = self.binerizer.transform([list(set(list(tagsets)
                                                        + updated_tagsets))])
                    finished = True
                except KeyError:
                    missing_tagset = sys.exc_info()[1].args[0]
                    updated_tagsets.remove(missing_tagset)

            Y[i] = new_row
        logging.info('Finished augmenting label mat with superclasses')
        return Y.toarray()

class FixedClassifierChain():

    def _init_classifier_chain(self, tagset_tree):
        chain_tree = dict()
        for head, tagset_branches in tagset_tree.items():
            chain_branches = list()
            for tagset_branch in tagset_branches:
                tagset = list(tagset_branch.keys())[0]
                chain_branch = self._init_classifier_chain(tagset_branch)
                chain_branches.append(chain_branch)
            chain_tree[head] = (deepcopy(self.base_classifier), chain_branches)
        return chain_tree

    def __init__(self, base_classifier, binerizer, \
                 subclass_dict=subclass_dict, tagset_tree=tagset_tree):
        self.subclass_dict = subclass_dict
        self.base_classifier = base_classifier
        self.binerizer = binerizer
        self.tagset_tree = tagset_tree
        self.classifier_chain = self._init_classifier_chain(self.tagset_tree)
        self.index_tagset_dict = dict([(tagset, i) for i, tagset \
                                       in enumerate(self.binerizer.classes_)])

    def _augment_labels_superclasses(self, Y):
        logging.info('Start augmenting label mat with superclasses')
        Y = lil_matrix(Y)
        for i, vect in enumerate(Y):
            tagsets = self.binerizer.inverse_transform(vect)[0]
            updated_tagsets = reduce(adder, [
                                find_keys(tagset, self.subclass_dict, check_in)
                                for tagset in tagsets], [])
            #TODO: This is bad code. need to be fixed later.
            finished = False
            while not finished:
                try:
                    new_row = self.binerizer.transform([list(set(list(tagsets)
                                                        + updated_tagsets))])
                    finished = True
                except KeyError:
                    missing_tagset = sys.exc_info()[1].args[0]
                    updated_tagsets.remove(missing_tagset)

            Y[i] = new_row
        logging.info('Finished augmenting label mat with superclasses')
        return Y

    def conv_array(self, d):
        if isinstance(d, np.ndarray):
            return d
        if isinstance(d, np.matrix):
            return np.asarray(d)
        else:
            return d.toarray()

    def conv_matrix(self, d):
        if isinstance(d, np.matrix):
            return d
        elif isinstance(d, np.ndarray):
            return np.matrix(d)
        else:
            return d.todense()

    def fit(self, X, Y, init_flag=True, classifier_chain=None, cnt=0):
        # Y and self.binerizer should be synchronized initially.
        if cnt%100==0:
            logging.info('{0}th step for prediction'.format(cnt))
        if init_flag:
            Y = self._augment_labels_superclasses(Y)
        assert Y.shape[0] == X.shape[0]
        cnt += 1
        if not classifier_chain:
            classifier_chain = self.classifier_chain
        for curr_head, (curr_classifier, branches) in classifier_chain.items():
            try:
                curr_head_index = self.index_tagset_dict[curr_head]
                y = self.conv_array(Y.T[curr_head_index])[0]
                try:
                    curr_classifier.fit(X, y) # TODO: Validate if this gets actually updated.
                except:
                    pdb.set_trace()
                branch_mask = np.where(y==1)[0]
                next_X = X[branch_mask]
                next_Y = Y[branch_mask]
                for branch in branches:
                    cnt = self.fit(next_X, next_Y, False, branch, cnt)
            except:
                pdb.set_trace()
        return cnt

    def _update_Y(self, new_y, label_index, indices):
        self.pred_Y[[indices],[label_index]] = new_y

    def predict(self, X, classifier_chain=None, given_mask='init', cnt=0):
        # Use branch_mask as a init flag too
        if cnt%100==0:
            logging.info('{0}th step for prediction'.format(cnt))
        if cnt==0:
            head_flag = True
        else:
            head_flag = False
        cnt += 1
        if given_mask == 'init':
            given_mask = np.array(range(0,X.shape[0]))
            # Init Y matrix
            self.pred_Y = np.zeros((X.shape[0], len(self.binerizer.classes_)))
        if not classifier_chain:
            classifier_chain = self.classifier_chain
        for curr_head, (curr_classifier, branches) in classifier_chain.items():
            sub_X = X[given_mask]
            if sub_X.shape[0]==0:
                continue
            pred_y = curr_classifier.predict(sub_X)
            #pred_y = curr_classifier.predict(sub_X.todense())
            label_index = self.index_tagset_dict[curr_head]
            self._update_Y(pred_y, label_index, given_mask)
            branch_mask = [given_mask[i] for i in np.where(pred_y==1)[0]]
            for branch in branches:
                cnt = self.predict(X, branch, branch_mask, cnt)
        if head_flag:
            return self.pred_Y
        else:
            return cnt

def tree_flatter(tree, init_flag=True):
    branches_list = list(tree.values())
    d_list = list(tree.keys())
    for branches in branches_list:
        for branch in branches:
            added_d_list = tree_flatter(branch)
            d_list = [d for d in d_list if d not in added_d_list]\
                    + added_d_list
    return d_list

def build_tagset_classifier(building_list, target_building,\
                            test_sentence_dict, test_token_label_dict,\
                            learning_phrase_dict, test_phrase_dict,\
                            learning_truths_dict,\
                            learning_srcids, test_srcids,\
                            tagset_list, eda_flag, use_brick_flag,
                            source_target_buildings,
                            n_jobs
                           ):
    learning_srcids = deepcopy(learning_srcids)
    global total_srcid_dict
    global point_tagsets
    global tagset_classifier_type 
#    source_target_buildings = list(set(building_list + [target_building]))
    #base_classifier = DecisionTreeClassifier()
    #base_classifier = SGDClassifier()
    base_classifier = LinearSVC(loss='hinge', tol=1e-5,\
                                max_iter=2000, C=2,
                                fit_intercept=False,
                                class_weight='balanced')

    #base_classifier = GaussianProcessClassifier()
#    base_classifier = QuadraticDiscriminantAnalysis()
    #base_classifier = GaussianNB()
    #base_classifier = SVC(kernel='rbf')
    #base_classifier = SVC(kernel='linear')
    #base_classifier = RandomForestClassifier(n_estimators=20)
    #base_classifier = AdaBoostClassifier()
#    clusterer = IGraphLabelCooccurenceClusterer('fastgreedy', weighted=True,
 #                                                   include_self_edges=True)
    #problem_transform_classifier = LabelPowerset(classifier=base_classifier)
    #tagset_classifier = LabelSpacePartitioningClassifier(problem_transform_classifier, clusterer)
    #tagset_classifier = LabelSpacePartitioningClassifier()
    #tagset_classifier = RakelO()
####    tagset_classifier = OneVsRestClassifier(SVC(kernel='rbf'), n_jobs=n_jobs)

   # tagset_lengther = lambda tagset: len(tagset.split('_'))
    #tagset_list = sorted(tagset_list, key=tagset_lengther)
    #TODO: Sort tagset_list based on the tree structure!
    new_tagset_list = tree_flatter(tagset_tree, [])
    new_tagset_list = new_tagset_list + [ts for ts in tagset_list \
                                     if ts not in new_tagset_list]
    tagset_list = new_tagset_list

    tagset_binerizer = MultiLabelBinarizer(tagset_list)
    tagset_binerizer.fit([tagset_list])
    assert tagset_list == tagset_binerizer.classes_.tolist()

    point_classifier = RandomForestClassifier(n_estimators=10, n_jobs=n_jobs)

    ## Init brick tag_list
    raw_tag_list = list(set(reduce(adder, map(splitter, tagset_list))))
    tag_list = deepcopy(raw_tag_list)


    # Extend tag_list with prefixes
    """
    if eda_flag:
        for building in set(building_list + [target_building]):
            prefixer = build_prefixer(building)
            building_tag_list = list(map(prefixer, raw_tag_list))
            tag_list = tag_list + building_tag_list
    """
    vocab_dict = dict([(tag, i) for i, tag in enumerate(tag_list)])

    proj_vectors = list()
    for tagset in tagset_list:
        tags = tagset.split('_')
        tags = tags + [building + '#' + tag for tag in tags \
                       for building in source_target_buildings]
        proj_vectors.append([1 if tag in tags else 0 for tag in tag_list])
    proj_vectors = np.asarray(proj_vectors)

    #tagset_classifier = custom_project_multi_label(base_classifier, proj_vectors)
    #tagset_classifier = FixedClassifierChain(base_classifier, tagset_binerizer)
    #tagset_classifier = BinaryRelevance(base_classifier)
    #tagset_classifier = ClassifierChain(base_classifier)
    #tagset_classifier = custom_multi_label(base_classifier)
    #tagset_classifier = DecisionTreeClassifier()
    #tagset_classifier = LabelPowerset(base_classifier)

    # Define Vectorizer
    tokenizer = lambda x: x.split()
    tagset_vectorizer = TfidfVectorizer(tokenizer=tokenizer,\
                                        vocabulary=vocab_dict)
    #tagset_vectorizer = CountVectorizer(tokenizer=tokenizer,\
    #                                    vocabulary=vocab_dict)

    ## Transform learning samples
    learning_doc = [' '.join(learning_phrase_dict[srcid]) \
                    for srcid in learning_srcids]
    test_doc = [' '.join(test_phrase_dict[srcid]) \
                for srcid in test_srcids]

    ## Augment with negative examples.
    negative_doc = []
    negative_truths_dict = {}
    negative_srcids = []
    for srcid in learning_srcids:
        true_tagsets = learning_truths_dict[srcid]
        sentence = learning_phrase_dict[srcid]
        for tagset in set(true_tagsets):
            negative_srcid = gen_uuid()
            negative_srcids.append(negative_srcid)
            negative_tagsets = list(filter(tagset.__ne__, true_tagsets))
            removing_tags = [word for word in sentence \
                                 if word in tagset.split('_')]
            negative_sentence = [ts for ts in true_tagsets\
                                 if not sum([tag in ts \
                                             for tag in removing_tags])]

#            negative_sentence = [word for word in sentence \
#                                 if word not in tagset.split('_')]
            negative_doc.append(' '.join(negative_sentence))
            negative_truths_dict[negative_srcid] = negative_tagsets
    for i in range(0,50):
        # Add empty examples
        negative_srcid = gen_uuid()
        negative_srcids.append(negative_srcid)
        negative_doc.append('')
        negative_truths_dict[negative_srcid] = []

    learning_doc += negative_doc
    learning_srcids += negative_srcids
    learning_truths_dict.update(negative_truths_dict)


    ## Init Brick document
    logging.info('Start adding Brick samples')
    if use_brick_flag:
        brick_copy_num = int(len(learning_phrase_dict) * 0.04)
        if brick_copy_num < 4:
            brick_copy_num = 4
        #brick_truths_dict = dict((gen_uuid(), [tagset]) \
        #                          for tagset in tagset_list\
        #                          for j in range(0, brick_copy_num))
        #for learning_srcid, true_tagsets in learning_truths_dict.items():
        #    for true_tagset in set(true_tagsets):
        #        brick_truths_dict[gen_uuid()] = [true_tagset]
#
        #brick_srcids = list(brick_truths_dict.keys())
        #brick_doc = [brick_truths_dict[tagset_id][0].replace('_', ' ')
        #                 for tagset_id in brick_srcids]
        brick_truths_dict = dict()
        brick_doc = list()
        brick_srcids = list()
        for tagset in tagset_list:
            for j in range(0, brick_copy_num):
                srcid = gen_uuid()
                brick_srcids.append(srcid)
                brick_truths_dict[srcid] = [tagset]
                brick_doc.append(' '.join(tagset.split('_') * ceil(j/2)))

        """
        if eda_flag:
            for building in set(building_list + [target_building]):
                for i in range(0, brick_copy_num):
                    for tagset in tagset_list:
                        brick_srcid = gen_uuid()
                        brick_srcids.append(brick_srcid)
                        brick_truths_dict[brick_srcid] = [tagset]
                        tags  = tagset.split('_') + \
                                [building + '#' + tag for tag in tagset.split('_')]
                        brick_doc.append(' '.join(tags))
        """
    else:
        brick_truths_dict = dict()
        brick_srcids = []
        brick_doc = []
    brick_truth_mat = csr_matrix([tagset_binerizer.transform(\
                                  [brick_truths_dict[srcid]])[0] \
                                  for srcid in brick_srcids])
    logging.info('Finished adding Brick samples')

    logging.info('start tagset vectorizing')
    tagset_vectorizer.fit(learning_doc + test_doc + brick_doc)
    logging.info('finished tagset vectorizing')

    if tagset_classifier_type == 'RandomForest':
        tagset_classifier = RandomForestClassifier(n_estimators=200, \
                                                   #this should be 100 at some point
                                                   random_state=0,\
                                                   n_jobs=n_jobs)
    elif tagset_classifier_type == 'StructuredCC':
        tagset_classifier = StructuredClassifierChain(base_classifier,
                                                      tagset_binerizer,
                                                      subclass_dict,
                                                      tagset_vectorizer.vocabulary,
                                                      n_jobs)
    else:
        assert False
    if eda_flag:
        unlabeled_phrase_dict = make_phrase_dict(\
                                    test_sentence_dict, \
                                    test_token_label_dict, \
                                    {target_building:test_srcids},\
                                    False)
        prefixer = build_prefixer(target_building)
        unlabeled_target_doc = [' '.join(\
                                map(prefixer, unlabeled_phrase_dict[srcid]))\
                                for srcid in test_srcids]
#        unlabeled_vect_doc = - tagset_vectorizer\
#                               .transform(unlabeled_target_doc)
        unlabeled_vect_doc = np.zeros((len(test_srcids), \
                                       len(tagset_vectorizer.vocabulary_)))
        test_doc = [' '.join(unlabeled_phrase_dict[srcid])\
                         for srcid in test_srcids]
        test_vect_doc = tagset_vectorizer.transform(test_doc).toarray()
        for building in source_target_buildings:
            if building == target_building:
                added_test_vect_doc = - test_vect_doc
            else:
                added_test_vect_doc = test_vect_doc
            unlabeled_vect_doc = np.hstack([unlabeled_vect_doc,\
                                            added_test_vect_doc])
    learning_doc += brick_doc
    learning_srcids += brick_srcids
    learning_truths_dict.update(brick_truths_dict)

    raw_learning_vect_doc = tagset_vectorizer.transform(learning_doc)
    learning_vect_doc = raw_learning_vect_doc.toarray()
    if eda_flag:
        for building in source_target_buildings:
            building_mask = np.array([1 if srcid in brick_srcids \
                                or find_key(srcid, total_srcid_dict, check_in)\
                                    == building\
                                        else 0 for srcid in learning_srcids])
            learning_vect_doc = np.hstack([learning_vect_doc] \
                                 + [np.asmatrix(building_mask \
                                    * learning_vect.toarray()[0]).T \
                                for learning_vect in raw_learning_vect_doc.T])

    truth_mat = csr_matrix([tagset_binerizer.transform(\
                    [learning_truths_dict[srcid]])[0]\
                        for srcid in learning_srcids])
    point_truths_dict = dict()
    point_srcids = list()
    for srcid, truths in learning_truths_dict.items():
        point_tagset = None
        for tagset in truths:
            if tagset in point_tagsets:
                point_tagset = tagset
                break
        if point_tagset:
            point_truths_dict[srcid] = point_tagset
            point_srcids.append(srcid)

    point_truth_mat = [point_tagsets.index(point_truths_dict[srcid]) \
                       for srcid in point_srcids]
    point_vect_doc = np.vstack([learning_vect_doc[learning_srcids.index(srcid)]
                                for srcid in point_srcids])
#    if use_brick_flag:
#        truth_mat = vstack([truth_mat, brick_truth_mat])
    if eda_flag:
        zero_vectors = tagset_binerizer.transform(\
                    [[] for i in range(0, unlabeled_vect_doc.shape[0])])
        for zero_vector in zero_vectors:
            truth_mat = vstack([truth_mat, zero_vector])
        #truth_mat = vstack([truth_mat] + zero_vectors)

#        truth_mat = truth_mat.toarray()

#        truth_mat = vstack([truth_mat, tagset_binerizer.transform(\
#                    [[] for i in range(0, unlabeled_vect_doc.shape[0])])])\
#                .toarray()
        learning_vect_doc = np.vstack([learning_vect_doc, unlabeled_vect_doc])
        #TODO: Check if learning_cet_Doct can be done without toarray()
    logging.info('Start learning multi-label classifier')
    tagset_classifier.fit(learning_vect_doc, truth_mat.toarray())
    logging.info('Finished learning multi-label classifier')
    #tagset_classifier.fit(coo_matrix(learning_vect_doc), truth_mat.toarray())
    point_classifier.fit(point_vect_doc, point_truth_mat)

    return tagset_classifier, tagset_vectorizer, tagset_binerizer, \
            point_classifier


def cross_validation(building_list, n_list,
                     target_building, \
                     learning_srcids, test_srcids, \
                     pred_phrase_dict, pred_tagsets_dict,
                     result_dict, k=2, \
                     eda_flag=False, token_type='justseparate'):

    learning_sentence_dict, \
    learning_token_label_dict, \
    learning_truths_dict, \
    learning_phrase_dict = get_multi_buildings_data(building_list, \
                                           learning_srcids, \
                                           eda_flag,\
                                           token_type)
    chosen_learning_srcids = random.sample(learning_srcids,\
                                           int(len(learning_srcids)/k))
#    validation_srcids = [srcid for srcid in learning_srcids \
#                         if srcid not in chosen_learning_srcids]
    validation_srcids = list()
    for building, n in zip(building_list, n_list):
        validation_srcids += select_random_samples(building, learning_srcids,\
                                                   n/k, use_cluster_flag=True)


    validation_sentence_dict, \
    validation_token_label_dict, \
    validation_truths_dict, \
    validation_phrase_dict = get_multi_buildings_data(building_list, \
                                           validation_srcids, \
                                           eda_flag,\
                                           token_type)
    test_sentence_dict,\
    test_token_label_dict,\
    _,\
    test_truths_dict,\
    _               = get_building_data(target_building, test_srcids,\
                                                eda_flag, token_type)


    cluster_dict = hier_clustering(pred_phrase_dict)
    cluster_dict = OrderedDict(\
        sorted(cluster_dict.items(), key=value_lengther, reverse=True))
    evaluated_srcids = list()
    precision_dict = dict()
    recall_dict = dict()
    validation_result_dict = dict()
    for cid, cluster_srcids in sorted(cluster_dict.items(), key=itemgetter(1)):
        building_list_1 = deepcopy(building_list)
        if target_building not in building_list_1:
            buliding_list_1.append(target_building)
        phrase_dict_1 = deepcopy(learning_phrase_dict)
        phrase_dict_1.update(pred_phrase_dict)
        truths_dict_1 = deepcopy(learning_truths_dict)
        truths_dict_1.update(pred_tagsets_dict)
        truths_dict_2 = deepcopy(learning_truths_dict)
        sentence_dict_2 = deepcopy(learning_sentence_dict)
        token_label_dict_2 = deepcopy(learning_token_label_dict)
        phrase_dict_2 = deepcopy(learning_phrase_dict)
        validation_building = building_list[0]


        try:
            tagset_classifier, tagset_vectorizer, tagset_binerizer, \
                    point_classifier = \
                build_tagset_classifier(building_list_1, validation_building,\
                                validation_sentence_dict, \
                                validation_token_label_dict,\
                                phrase_dict_1, phrase_dict_2,\
                                truths_dict_1,\
                                chosen_learning_srcids+cluster_srcids, \
                                validation_srcids,\
                                tagset_list, eda_flag
                               )
        except:
            pdb.set_trace()

        validation_pred_tagsets_dict, validation_pred_certainty_dict = \
                tagsets_prediction(tagset_classifier, tagset_vectorizer, \
                                   tagset_binerizer, validation_phrase_dict, \
                                   validation_srcids)
        curr_result_dict = tagsets_evaluation(validation_truths_dict, \
                                         validation_pred_tagsets_dict, \
                                         validation_pred_certainty_dict,\
                                         validation_srcids)

        precision_dict[cid] = curr_result_dict['point_precision']
        recall_dict[cid] = curr_result_dict['point_recall']
        # Finding found points in target building
        true_found_points = list()
        pred_found_points = list()
        for srcid in cluster_srcids:
            true_tagsets = test_truths_dict[srcid]
            pred_tagsets = pred_tagsets_dict[srcid]
            for tagset in true_tagsets:
                if tagset in point_tagsets:
                    true_found_points.append(tagset)
            for tagset in pred_tagsets:
                if tagset in point_tagsets:
                    pred_found_points.append(tagset)

        validation_result_dict[cid] = {
            'srcids': cluster_srcids,
            'common_pred_points': pred_found_points,
            'common_true_points': true_found_points,
            'point_precision': precision_dict[cid],
            'point_recall': recall_dict[cid]
        }
        if False: #success: #TODO: Implement this.
            evaluated_srcids += cluster_srcids
    with open('result/validation_result.json', 'w') as fp:
        json.dump(validation_result_dict, fp, indent=2)

    cluster_precision_list = [c['point_precision'] for c \
                              in validation_result_dict.values()]
    for cid, cluster_result in validation_result_dict.items():
        # TODO: Think about better evaludation metric for this.
        try:
            if cluster_result['point_precision'] > \
                    np.mean(cluster_precision_list)\
               and len(cluster_result['common_pred_points']) > 1:
                learning_srcids += cluster_result['srcids']
        except:
            pdb.set_trace()
    return learning_srcids


def certainty_getter(x):
    return x[1]['certainty']


def extend_tagset_list(new_tagsets):
    global tagset_list
    tagset_list.extend(new_tagsets)
    tagset_list = list(set(tagset_list))

def entity_recognition_from_ground_truth(building_list,
                                         source_sample_num_list,
                                         target_building,
                                         token_type='justseparate',
                                         label_type='label',
                                         use_cluster_flag=False,
                                         use_brick_flag=False,
                                         debug_flag=False,
                                         eda_flag=False,
                                         n_jobs=1,
                                         prev_step_data={
                                             'learning_srcids':[],
                                             'iter_cnt':0,
                                             'point_precision_history': [],
                                             'point_recall_history':  [],
                                             'correct_point_cnt_history': [],
                                             'incorrect_point_cnt_history': [],
                                             'unfound_point_cnt_history': [],
                                         },
                                         crf_phrase_dict=None,
                                         crf_srcids=None
                                        ):
    logging.info('Entity Recognition Get Started.')
    global tagset_list
    global total_srcid_dict
    assert len(building_list) == len(source_sample_num_list)

    ########################## DATA INITIATION ##########################
    # construct source data information data structure
    source_cnt_list = [[building, cnt]\
                       for building, cnt\
                       in zip(building_list, source_sample_num_list)]

    # Read previous step's data
    learning_srcids = prev_step_data.get('learning_srcids')
    prev_test_srcids = prev_step_data.get('test_srcids')
    prev_pred_tagsets_dict = prev_step_data.get('pred_tagsets_dict')
    prev_pred_certainty_dict = prev_step_data.get('pred_certainty_dict')
    prev_pred_phrase_dict = prev_step_data.get('pred_phrase_dict')
    prev_result_dict = prev_step_data.get('result_dict')
    prev_iter_cnt = prev_step_data.get('iter_cnt')
    iter_cnt = prev_step_data['iter_cnt'] + 1


    if iter_cnt>1:
        learning_srcids = cross_validation(building_list, source_sample_num_list, \
                         target_building, \
                         learning_srcids, prev_test_srcids, \
                         prev_pred_phrase_dict, prev_pred_tagsets_dict,
                         prev_result_dict, 4, \
                         eda_flag, token_type)
    print('\n')
    print('################ Iteration {0} ################'.format(iter_cnt))

    ### Learning Data
    sample_srcid_list_dict = dict()
    for building, sample_num in zip(building_list, source_sample_num_list):
        with open('metadata/{0}_char_label_dict.json'\
                  .format(building), 'r') as fp:
            sentence_label_dict = json.load(fp)
        if iter_cnt == 1:
            sample_srcid_list = select_random_samples(\
                                    building,\
                                    sentence_label_dict.keys(),\
                                    sample_num, \
                                    use_cluster_flag,\
                                    token_type=token_type)
            sample_srcid_list_dict[building] = sample_srcid_list
            learning_srcids += sample_srcid_list
            total_srcid_dict[building] = list(sentence_label_dict.keys())
        else:
            sample_srcid_list_dict[building] = [srcid for srcid\
                                                in sentence_label_dict.keys()\
                                                if srcid in learning_srcids]
    learning_sentence_dict, \
    learning_token_label_dict, \
    learning_truths_dict, \
    phrase_dict = get_multi_buildings_data(building_list, \
                                           learning_srcids, \
                                           eda_flag,\
                                           token_type)
    found_points = [tagset for tagset \
                     in reduce(adder, learning_truths_dict.values(), []) \
                             if tagset in point_tagsets]
    found_point_cnt_dict = Counter(found_points)
    found_points = set(found_points)

    ### Get Test Data
    with open('metadata/{0}_char_label_dict.json'\
              .format(target_building), 'r') as fp:
        sentence_label_dict = json.load(fp)
    test_srcids = [srcid for srcid in sentence_label_dict.keys() \
                       if srcid not in learning_srcids]
    total_srcid_dict[target_building] = list(sentence_label_dict.keys())
    test_sentence_dict,\
    test_token_label_dict,\
    test_phrase_dict,\
    test_truths_dict,\
    test_srcid_dict = get_building_data(target_building, test_srcids,\
                                                eda_flag, token_type)


    extend_tagset_list(reduce(adder, \
                [learning_truths_dict[srcid] for srcid in learning_srcids]\
                + [test_truths_dict[srcid] for srcid in test_srcids], []))
    augment_tagset_tree(tagset_list)

    source_target_buildings = list(set(building_list + [target_building]))
    begin_time = arrow.get()
    tagset_classifier, tagset_vectorizer, tagset_binerizer, \
            point_classifier = \
            build_tagset_classifier(building_list, target_building,\
            #                learning_sentence_dict, ,\
                            test_sentence_dict,\
            #                learning_token_label_dict,\
                            test_token_label_dict,\
                            phrase_dict, test_phrase_dict,\
                            learning_truths_dict,\
                            learning_srcids, test_srcids,\
                            tagset_list, eda_flag, use_brick_flag,\
                            source_target_buildings,
                            n_jobs
                           )
    end_time = arrow.get()
    print('Training Time: {0}'.format(end_time-begin_time))

    ###########################  LEARNING  ################################

    ## Brick Validation
    brick_doc = []
    if use_brick_flag:
        brick_phrase_dict = dict([(str(i), tagset.split('_')) for i, tagset\
                                  in enumerate(tagset_list)])
        brick_srcids = list(brick_phrase_dict.keys())
        brick_pred_tagsets_dict, brick_pred_certainty_dict, \
                brick_pred_point_dict = \
                tagsets_prediction(tagset_classifier, tagset_vectorizer, \
                                   tagset_binerizer, \
                                   brick_phrase_dict, \
                                   list(brick_phrase_dict.keys()),
                                   source_target_buildings,
                                   eda_flag,
                                   point_classifier
                                  )

    ####################      TEST      #################
    ### Self Evaluation

    learning_pred_tagsets_dict, \
    learning_pred_certainty_dict, \
    learning_pred_point_dict = tagsets_prediction(\
                                tagset_classifier, \
                                tagset_vectorizer, \
                                tagset_binerizer, \
                                phrase_dict, \
                                sorted(learning_srcids),\
                                building_list,\
                                eda_flag,\
                                point_classifier)
    pdb.set_trace()
    learning_result_dict = tagsets_evaluation(learning_truths_dict, \
                                         learning_pred_tagsets_dict, \
                                         learning_pred_certainty_dict,\
                                         sorted(learning_srcids),\
                                         learning_pred_point_dict,\
                                         phrase_dict,\
                                         debug_flag)
    #TODO: Test below and remove if not necessary
#    tagset_vectorizer.fit(test_doc)
    """
    test_doc = [' '.join(test_phrase_dict[srcid]) for srcid in test_srcids]
    if eda_flag:
        test_vect_doc = eda_vectorizer(tagset_vectorizer, test_doc, \
                                       source_target_buildings, test_srcids)
    else:
        test_vect_doc = tagset_vectorizer.transform(test_doc)

    pred_certainty_dict = dict()
    pred_tagsets_dict = dict()
    pred_mat = tagset_classifier.predict(test_vect_doc)
    #prob_mat = tagset_classifier.predict_proba(test_vect_doc)
    for i, (srcid, pred) in enumerate(zip(test_srcids, pred_mat)):
        pred_tagsets_dict[srcid] = tagset_binerizer.inverse_transform(\
                                        np.asarray([pred]))[0]
        #pred_tagsets_dict[srcid] = list(tagset_binerizer\
        #                                .inverse_transform(pred)[0])
        # TODO: Don't remove below. Activate this when using RandomForest
        #pred_vec = [prob[i][0] for prob in prob_mat]
        #pred_certainty_dict[srcid] = sum(pred_vec) / float(len(pred)-sum(pred))
        pred_certainty_dict[srcid] = 0
    pred_certainty_dict = OrderedDict(sorted(pred_certainty_dict.items(), \
                                             key=itemgetter(1), reverse=True))

    ############## EVALUATE TESTS #############
    # Evaluate result 
    # TODO: Check if fault predictions are related to unincluded point tagsets
    found_point_tagsets = set([tagset for tagset \
                               in reduce(adder, \
                                         learning_truths_dict.values(), \
                                         [])])
    # regenerate learning_doc and learning_vect_doc for debugging
    result_dict = defaultdict(dict)

    sorted_result_dict = OrderedDict()
    incorrect_tagsets_dict = dict()
    correct_cnt = 0
    incorrect_cnt = 0
    point_correct_cnt = 0
    point_incorrect_cnt = 0
    empty_point_cnt = 0
    unknown_reason_cnt = 0
    undiscovered_point_cnt = 0
    unfound_points = set()
    for srcid, pred_tagsets in pred_tagsets_dict.items():
        true_tagsets = test_truths_dict[srcid]
        one_result = {
            'tagsets': pred_tagsets,
            'certainty': pred_certainty_dict[srcid]
        }
        if set(true_tagsets) == set(pred_tagsets):
            correct_cnt += 1
            one_result['correct?'] = True
            result_dict['correct'][srcid] = one_result
            #result_dict['correct'][srcid] = pred_tagsets
            point_correct_cnt += 1
        else:
            incorrect_cnt += 1
            one_result['correct?'] = False
            result_dict['incorrect'][srcid] = one_result
            true_point = None
            for tagset in true_tagsets:
                if tagset in point_tagsets:
                    true_point = tagset
                    break
            try:
                assert true_point
                if true_point not in found_points:
                    unfound_points.add(true_point)
                if true_point not in found_points or true_point=='unknown':
                    undiscovered_point_cnt += 1
                found_point = None
                for tagset in pred_tagsets:
                    if tagset in point_tagsets:
                        found_point = tagset
                        break
                if not found_point:
                    empty_point_cnt += 1
                elif found_point != true_point:
                    point_incorrect_cnt += 1
                    print("INCORRECT POINT FOUND: {0} -> {1}"\
                          .format(true_point, found_point))
                else:
                    unknown_reason_cnt += 1
                    point_correct_cnt += 1
                #    pdb.set_trace()
            except:
                print('point not found')
                pdb.set_trace()
                unknown_reason_cnt += 1
            if debug_flag:
                print('####################################################')
                print('TRUE: {0}'.format(true_tagsets))
                print('PRED: {0}'.format(pred_tagsets))
                if true_point:
                    print('point num in source building: {0}'\
                          .format(found_point_cnt_dict[true_point]))
                else:
                    print('no point is included here')
                source_srcid = None
                source_idx = None
                for temp_srcid, tagsets in learning_truths_dict.items():
                    if true_point and true_point in tagsets:
                        #source_srcid = temp_srcid
                        #source_idx = learning_srcids.index(source_srcid)
                        #source_doc = learning_doc[source_idx]
                        #source_vect = learning_vect_doc[source_idx]
                        break
                test_idx = test_srcids.index(srcid)
                target_doc = test_doc[test_idx]
                target_vect = test_vect_doc[test_idx]
                print('####################################################')
                if not found_point and true_point in found_points\
                   and true_point not in ['unknown',\
                        'effective_heating_temperature_setpoint',\
                        'effective_cooling_temperature_setpoint',\
                        'supply_air_flow_setpoint',\
                        'output_frequency_sensor']:
                    source_point_srcid = None
                    for temp_srcid, truths in learning_truths_dict.items():
                        if true_point in truths:
                            source_point_srcid = temp_srcid
                            break
                    print('--source phrase: {0}'.format(\
                                Counter(phrase_dict[source_point_srcid])))
                    print('--target phrase: {0}'.format(\
                                Counter(test_phrase_dict[srcid])))

                    pdb.set_trace()
                    pass
        sorted_result_dict[srcid] = one_result

    point_precision = float(point_correct_cnt) \
                        / (point_correct_cnt + point_incorrect_cnt)
    point_recall = float(point_correct_cnt) \
                        / (point_correct_cnt + empty_point_cnt)
    print('------------------------------------result---------------')
    print('point precision: {0}'.format(point_precision))
    print('point recall: {0}'.format(point_recall))
    print(len(unfound_points))
    print(len(found_points))
    if empty_point_cnt > 0:
        print('rate points not found in source \
              among sensors where point is not found: \n\t{0}'\
              .format(undiscovered_point_cnt / float(empty_point_cnt)))
    print('sensors where a point is not found: ', empty_point_cnt\
                                               /float(incorrect_cnt),\
                                empty_point_cnt)
    print('sensors where incorrect points are found: ', point_incorrect_cnt\
                                                     /float(incorrect_cnt),\
                                      point_incorrect_cnt)
    print('unknown reason: ', unknown_reason_cnt\
                              /float(incorrect_cnt),\
                              unknown_reason_cnt)
    print('-----------')


    sorted_result_dict = OrderedDict(\
                            sorted(sorted_result_dict.items(), \
                                   key=certainty_getter))
    sorted_result_dict['samples'] = learning_srcids
    result_dict['samples'] = learning_srcids

    print('precision')
    print(float(correct_cnt) / len(test_srcids))
    with open('result/tagset_{0}.json'.format(building), 'w') as fp:
        json.dump(result_dict, fp, indent=2)
    with open('result/sorted_tagset_{0}.json'.format(building), 'w') as fp:
        json.dump(sorted_result_dict, fp, indent=2)
    """

    ### Test on the entire target building
    target_srcids = raw_srcids_dict[target_building]
    _,\
    _,\
    target_phrase_dict,\
    target_truths_dict,\
    _                   = get_building_data(target_building, \
                                            target_srcids,\
                                            eda_flag, token_type)
    target_pred_tagsets_dict, \
    target_pred_certainty_dict, \
    target_pred_point_dict = tagsets_prediction(\
                                tagset_classifier, \
                                tagset_vectorizer, \
                                tagset_binerizer, \
                                target_phrase_dict, \
                                target_srcids,\
                                source_target_buildings,\
                                eda_flag,\
                                point_classifier)
    target_result_dict = tagsets_evaluation(target_truths_dict, \
                                         target_pred_tagsets_dict, \
                                         target_pred_certainty_dict,\
                                         target_srcids,\
                                         target_pred_point_dict,\
                                         target_phrase_dict,\
                                            debug_flag)
    next_step_data = {
        'pred_tagsets_dict': target_pred_tagsets_dict,
        'learning_srcids': learning_srcids,
        'test_srcids': test_srcids,
#        'pred_certainty_dict': pred_certainty_dict,
        'iter_cnt': iter_cnt,
    #    'result_dict': result_dict,
        'pred_phrase_dict': test_phrase_dict,
        'point_precision_history': \
            prev_step_data['point_precision_history'] \
            + [target_result_dict['point_precision']],
        'point_recall_history': \
            prev_step_data['point_recall_history'] \
            + [target_result_dict['point_recall']],
        'correct_point_cnt_history': \
            prev_step_data['correct_point_cnt_history'] \
            + [target_result_dict['point_correct_cnt']],
        'incorrect_point_cnt_history': \
            prev_step_data['incorrect_point_cnt_history'] \
            + [target_result_dict['point_incorrect_cnt']],
        'unfound_point_cnt_history': \
            prev_step_data['unfound_point_cnt_history'] \
            + [target_result_dict['unfound_point_cnt']]
    }

    #pdb.set_trace()
    print('################################# Iter# {0}'.format(iter_cnt))
    print('history of point precision: {0}'\
          .format(next_step_data['point_precision_history']))
    print('history of point recall: {0}'\
          .format(next_step_data['point_recall_history']))
    print('history of correct point cnt: {0}'\
          .format(next_step_data['correct_point_cnt_history']))
    print('history of incorrect point cnt: {0}'\
          .format(next_step_data['incorrect_point_cnt_history']))
    print('history of unfound point cnt: {0}'\
          .format(next_step_data['unfound_point_cnt_history']))

    return  target_result_dict['point_precision'], \
            target_result_dict['point_recall'], \
            next_step_data


def make_ontology():
    pass


def parallel_func(orig_func, return_idx, return_dict, *args):
    return_dict[return_idx] = orig_func(*args)


def entity_recognition_iteration(iter_num, *args):
    step_data={
        'learning_srcids':[],
        'iter_cnt':0,
        'point_precision_history': [],
        'point_recall_history':  [],
        'correct_point_cnt_history': [],
        'incorrect_point_cnt_history': [],
        'unfound_point_cnt_history': [],
    }
    for i in range(0, iter_num):
        _, _, step_data = entity_recognition_from_ground_truth(\
                              building_list = args[0],\
                              source_sample_num_list = args[1],\
                              target_building = args[2],\
                              token_type = args[3],\
                              label_type = args[4],\
                              use_cluster_flag = args[5],\
                              use_brick_flag = args[6],\
                              debug_flag = args[7],\
                              eda_flag = args[8],\
                              n_jobs = args[9],\
                              prev_step_data = step_data
                            )


def determine_used_tokens(sentence, token_labels, tagsets):
    token_usages = list()
    tags = reduce(adder, [tagset.split('_') for tagset in tagsets], [])
    for token, label in zip(sentence, token_labels):
        if label=='O':
            token_usages.append(0)
        else:
            tags_in_label = label[2:].split('_')
            included_cnt = 0
            for tag in tags_in_label:
                if tag in tags:
                    included_cnt += 1
            token_usages.append(included_cnt / len(tags_in_label))

    return token_usages

def determine_used_tokens_multiple(sentence_dict, token_label_dict, \
                                   tagsets_dict, srcids):
    token_usage_dict = dict()
    for srcid in srcids:
        token_usage_dict[srcid] = determine_used_tokens(\
                                        sentence_dict[srcid],\
                                        token_label_dict[srcid],\
                                        tagsets_dict[srcid])
    return token_usage_dict



def entity_recognition_from_crf(building_list,\
                                source_sample_num_list,\
                                target_building,\
                                token_type='justseparate',\
                                label_type='label',\
                                use_cluster_flag=False,\
                                use_brick_flag=False,\
                                eda_flag=False,\
                                debug_flag=False):

    global tagset_list
    global total_srcid_dict

    ### Initialize CRF Data
    crf_result_query = {
        'label_type': label_type,
        'token_type': token_type,
        'use_cluster_flag': use_cluster_flag,
        'building_list': building_list,
        'source_sample_num_list': source_sample_num_list,
        'target_building': target_building
    }
    crf_result = get_crf_results(crf_result_query)
    given_srcids = list(reduce(adder,\
                            list(crf_result['source_list'].values()), []))
    crf_sentence_dict = dict()
    crf_token_label_dict = {}
    for srcid, one_crf_result in crf_result['result'].items():
        crf_token_label_dict[srcid] = one_crf_result['pred_token_labels']
        crf_sentence_dict[srcid] = one_crf_result['sentence']
    crf_phrase_dict = make_phrase_dict(crf_sentence_dict,\
                                       crf_token_label_dict, \
                                       {target_building: \
                                        crf_token_label_dict.keys()})
    crf_srcids = [srcid for srcid in crf_sentence_dict.keys() \
                  if srcid not in given_srcids]

    ### Initialize Given (Learning) Data
    _, \
    _, \
    _, \
    crf_truths_dict, \
    _                 = get_building_data(target_building, crf_srcids, \
                                        eda_flag, token_type)
    given_sentence_dict, \
    given_token_label_dict, \
    given_truths_dict, \
    given_phrase_dict = get_multi_buildings_data(\
                                building_list, given_srcids, \
                                eda_flag, token_type)
    extend_tagset_list(reduce(adder, \
                [given_truths_dict[srcid] for srcid in given_srcids]\
                + [crf_truths_dict[srcid] for srcid in crf_srcids], []))
    classifier, vectorizer, binerizer, point_classifier = \
            build_tagset_classifier(building_list, target_building,\
                                    crf_sentence_dict, crf_token_label_dict,\
                                    given_phrase_dict, crf_phrase_dict,\
                                    given_truths_dict,\
                                    given_srcids, crf_srcids,\
                                    tagset_list, eda_flag, use_brick_flag
                                   )
    #augment_tagset_tree(tagset_list)
    #pdb.set_trace()

    crf_pred_tagsets_dict, crf_pred_certainty_dict = \
                tagsets_prediction(classifier, vectorizer, \
                                   binerizer, crf_phrase_dict, \
                                   crf_srcids)
    crf_token_usage_dict = determine_used_tokens_multiple(\
                                crf_sentence_dict, crf_token_label_dict, \
                                crf_pred_tagsets_dict, crf_srcids)

    crf_token_usage_rate_dict = dict((srcid, sum(usage)/len(usage))\
                                     for srcid, usage \
                                     in crf_token_usage_dict.items())


    pdb.set_trace()



#TODO: Make this more generic to apply to other functions
def entity_recognition_from_ground_truth_get_avg(N,
                                                 building_list,
                                                 source_sample_num_list,
                                                 target_building,
                                                 token_type='justseparate',
                                                 label_type='label',
                                                 use_cluster_flag=False,
                                                 use_brick_flag=False,
                                                 eda_flag=True):
    worker_num = 1

    manager = Manager()
    return_dict = manager.dict()
    jobs = []

    args = (building_list,\
            source_sample_num_list,\
            target_building,\
            token_type,
            label_type,
            use_cluster_flag,
            use_brick_flag,
            False,
            eda_flag)

    for i in range(0,N):
        p = Process(target=parallel_func, args=(\
                entity_recognition_from_ground_truth, i, return_dict, *args))
#                                               ))
        jobs.append(p)
        p.start()
        if i % worker_num == worker_num-1:
            for proc in jobs:
                proc.join()
            jobs = []

    for proc in jobs:
        proc.join()

    avg_prec = np.mean(list(map(itemgetter(0), return_dict.values())))
    avg_recall  = np.mean(list(map(itemgetter(1), return_dict.values())))
    print('=======================================================')
    print ('Averaged Point Precision: {0}'.format(avg_prec))
    print ('Averaged Point Recall: {0}'.format(avg_recall))

    print("FIN")


def str2bool(v):
    if v in ['true', 'True']:
        return True
    elif v in ['false', 'False']:
        return False
    else:
        assert(False)

def str2slist(s):
    s.replace(' ', '')
    return s.split(',')

def str2ilist(s):
    s.replace(' ', '')
    return [int(c) for c in s.split(',')]

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)
    parser.register('type','slist', str2slist)
    parser.register('type','ilist', str2ilist)

    parser.add_argument(choices=['learn', 'predict', 'entity', 'crf_entity', \
                                 'init'],
                        dest = 'prog')

    parser.add_argument('predict',
                         action='store_true',
                         default=False)

    """
    parser.add_argument('-b',
                        type=str,
                        help='Learning source building name',
                        dest='source_building')
    parser.add_argument('-n', 
                        type=int, 
                        help='The number of learning sample',
                        dest='sample_num')
    """

    parser.add_argument('-bl',
                        type='slist',
                        help='Learning source building name list',
                        dest='source_building_list')
    parser.add_argument('-nl',
                        type='ilist',
                        help='A list of the number of learning sample',
                        dest='sample_num_list')
    parser.add_argument('-l',
                        type=str,
                        help='Label type (either label or category',
                        default='label',
                        dest='label_type')
    parser.add_argument('-c',
                        type='bool',
                        help='flag to indicate use hierarchical cluster \
                                to select learning samples.',
                        default=False,
                        dest='use_cluster_flag')
    parser.add_argument('-d',
                        type='bool',
                        help='Debug mode flag',
                        default=False,
                        dest='debug_flag')
    parser.add_argument('-t',
                        type=str,
                        help='Target buildling name',
                        dest='target_building')
    parser.add_argument('-eda',
                        type='bool',
                        help='Flag to use Easy Domain Adapatation',
                        default=False,
                        dest='eda_flag')
    parser.add_argument('-ub',
                        type='bool',
                        help='Use Brick when learning',
                        default=False,
                        dest='use_brick_flag')
    parser.add_argument('-avg',
                        type=int,
                        help='Number of exp to get avg. If 1, ran once',
                        dest='avgnum',
                        default=1)
    parser.add_argument('-iter',
                        type=int,
                        help='Number of iteration for the given work',
                        dest='iter_num',
                        default=1)
    parser.add_argument('-nj',
                        type=int,
                        help='Number of processes for multiprocessing',
                        dest='n_jobs',
                        default=1)
    parser.add_argument('-ct',
                        type=str,
                        help='Tagset classifier type. one of RandomForest, \
                              StructuredCC.',
                        dest='tagset_classifier_type')

    args = parser.parse_args()

    tagset_classifier_type = args.tagset_classifier_type

    if args.prog == 'learn':
        learn_crf_model(building_list=args.source_building_list,
                        source_sample_num_list=args.sample_num_list,
                        token_type='justseparate',
                        label_type=args.label_type,
                        use_cluster_flag=args.use_cluster_flag,
                        debug_flag=args.debug_flag,
                        use_brick_flag=args.use_brick_flag)
    elif args.prog == 'predict':
        crf_test(building_list=args.source_building_list,
                 source_sample_num_list=args.sample_num_list,
                 target_building=args.target_building,
                 token_type='justseparate',
                 label_type=args.label_type,
                 use_cluster_flag=args.use_cluster_flag,
                 use_brick_flag=args.use_brick_flag)
    elif args.prog == 'entity':
        if args.avgnum == 1:
            entity_recognition_iteration(args.iter_num,
                                         args.source_building_list,
                                         args.sample_num_list,
                                         args.target_building,
                                         'justseparate',
                                         args.label_type,
                                         args.use_cluster_flag,
                                         args.use_brick_flag,
                                         args.debug_flag,
                                         args.eda_flag,
                                         args.n_jobs
                                        )
            """
            entity_recognition_from_ground_truth(\
                building_list=args.source_building_list,
                source_sample_num_list=args.sample_num_list,
                target_building=args.target_building,
                token_type='justseparate',
                label_type=args.label_type,
                use_cluster_flag=args.use_cluster_flag,
                use_brick_flag=args.use_brick_flag,
                debug_flag=args.debug_flag,
                eda_flag=args.eda_flag)
            """
        elif args.avgnum>1:
            entity_recognition_from_ground_truth_get_avg(args.avgnum,
                building_list=args.source_building_list,
                source_sample_num_list=args.sample_num_list,
                target_building=args.target_building,
                token_type='justseparate',
                label_type=args.label_type,
                use_cluster_flag=args.use_cluster_flag,
                use_brick_flag=args.use_brick_flag)
    elif args.prog == 'crf_entity':
        entity_recognition_from_crf(\
                building_list=args.source_building_list,\
                source_sample_num_list=args.sample_num_list,\
                target_building=args.target_building,\
                token_type='justseparate',\
                label_type=args.label_type,\
                use_cluster_flag=args.use_cluster_flag,\
                use_brick_flag=args.use_brick_flag,\
                eda_flag=args.eda_flag,
                debug_flag=args.debug_flag)
    elif args.prog == 'init':
        init()
    else:
        #print('Either learn or predict should be provided')
        assert(False)
