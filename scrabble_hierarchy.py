from functools import reduce, partial
import os
import json
import random
from collections import OrderedDict, defaultdict, Counter
import pdb
from copy import deepcopy
from operator import itemgetter
from itertools import islice, chain
import argparse
import logging
from imp import reload
from uuid import uuid4
def gen_uuid():
    return str(uuid4())
from multiprocessing import Pool, Manager, Process
import code
import re
import sys
from math import ceil, floor
from pprint import PrettyPrinter
pp = PrettyPrinter()
import psutil
import pickle
import subprocess

import pycrfsuite
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
#plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.labelpad'] = 0
plt.rcParams['font.size'] = 8
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.axes import Axes
from bson.binary import Binary as BsonBinary
import arrow
from pygame import mixer
import pympler

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,\
                             GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, chi2, SelectPercentile, SelectKBest
from sklearn.pipeline import Pipeline
from scipy.sparse import vstack, csr_matrix, hstack, issparse, coo_matrix,\
                         lil_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neural_network import MLPClassifier
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as hier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from skmultilearn.ensemble import RakelO, LabelSpacePartitioningClassifier
from skmultilearn.problem_transform import LabelPowerset, ClassifierChain, \
                                           BinaryRelevance
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
#from skmultilearn.cluster import IGraphLabelCooccurenceClusterer
#from skmultilearn.ensemble import LabelSpacePartitioningClassifier

import plotter
from resulter import Resulter
from time_series_to_ir import TimeSeriesToIR
from mongo_models import store_model, get_model, get_tags_mapping, \
                         get_crf_results, store_result, get_entity_results
from brick_parser import pointTagsetList        as  point_tagsets,\
                         locationTagsetList     as  location_tagsets,\
                         equipTagsetList        as  equip_tagsets,\
                         pointSubclassDict      as  point_subclass_dict,\
                         equipSubclassDict      as  equip_subclass_dict,\
                         locationSubclassDict   as  location_subclass_dict,\
                         tagsetTree             as  tagset_tree
#                         tagsetList as tagset_list
tagset_tree['networkadapter'] = list()
tagset_tree['unknown'] = list()
tagset_tree['none'] = list()
subclass_dict = dict()
subclass_dict.update(point_subclass_dict)
subclass_dict.update(equip_subclass_dict)
subclass_dict.update(location_subclass_dict)
subclass_dict['networkadapter'] = list()
subclass_dict['unknown'] = list()
subclass_dict['none'] = list()
tagset_classifier_type = None
ts_feature_filename = 'TS_Features/features.pkl'

from building_tokenizer import nae_dict


anon_building_dict = {
    'ebu3b': 'A-1',
    'bml': 'A-2',
    'ap_m': 'A-3',
    'ghc': 'B-1'
}

point_tagsets += [#'unknown', \
                  #'run_command', \
                  #'low_outside_air_temperature_enable_differential_setpoint', \
                  #'co2_differential_setpoint', 
                  #'pump_flow_status', \
                  #'supply_air_temperature_increase_decrease_step_setpoint',\
                  #'average_exhaust_air_static_pressure_setpoint', \
                  #'chilled_water_differential_pressure_load_shed_command', \
                  # AP_M
                 # 'average_exhaust_air_pressure_setpoint', 
                 # 'chilled_water_temperature_differential_setpoint',
                 # 'outside_air_lockout_temperature_differential_setpoint',
                 # 'vfd_command',
                 ] #TODO: How to add these into the tree structure?

tagset_list = point_tagsets + location_tagsets + equip_tagsets
tagset_list.append('networkadapter')

for tagset in tagset_list:
    if tagset.split('_')[-1]=='shed':
        pdb.set_trace()

total_srcid_dict = dict()
tree_depth_dict = dict()

logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s')

def init_srcids_dict():
    building_list = ['ebu3b', 'ap_m', 'bml', 'ghc']
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

def calc_leaves_depth(tree, d=dict(), depth=0):
    curr_depth = depth + 1
    for tagset, branches in tree.items():
        if d.get(tagset):
            d[tagset] = max(d[tagset], curr_depth)
        else:
            d[tagset] = curr_depth
        for branch in branches:
            new_d = calc_leaves_depth(branch, d, curr_depth)
            for k, v in new_d.items():
                if d.get(k):
                    d[k] = max(d[k], v)
                else:
                    d[k] = v
    return d


class TsTfidfVectorizer():
    def __init__(self, vocabulary=None):
        self.count_vectorizer = CountVectorizer(vocabulary=vocabulary)
        self.ts2ir = None

    def fit(self, X, srcids):
        new_X = self.CountVectorizer.fit_transform(X)
        with open("Binarizer/mlb.pkl", 'rb') as f:
            mlb = pickle.load(f, encoding='bytes')
        with open("TS_Features/ebu3b_features.pkl", 'rb') as f:
            ts_features = pickle.load(f)
        self.ts2ir.train_model(ts_features, srcids)
        mlb_keys, Y_pred, Y_proba = self.ts2ir.kpredict(ts_features, srcids)
        pdb.set_trace()

    def transform(self, X):
        pass

    def add_words_from_ts(learning_srcids, target_srcids):
        with open("Binarizer/mlb.pkl", 'rb') as f:
            mlb = pickle.load(f, encoding='bytes')
        with open("TS_Features/ebu3b_features.pkl", 'rb') as f:
            ts_features = pickle.load(f)
        ts2ir = TimeSeriesToIR(mlb=mlb)
        mlb_keys, Y_pred, Y_proba =\
                ts2ir.ts_to_ir(ts_features, learning_srcids, test_srcids)
        pdb.set_trace()


def play_end_alarm():
    mixer.init()
    mixer.music.load('etc/fins_success.wav')
    mixer.music.play()

def adder(x, y):
    return x+y

def splitter(s):
    return s.split('_')

def alpha_tokenizer(s): 
    return re.findall('[a-zA-Z]+', s)

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


def subset_accuracy_func(pred_Y, true_Y, labels):
    return 1 if set(pred_Y) == set(true_Y) else 0

def get_score(pred_dict, true_dict, srcids, score_func, labels):
    score = 0
    for srcid in srcids:
        pred_tagsets = pred_dict[srcid]
        true_tagsets = true_dict[srcid]
        if isinstance(pred_tagsets, list):
            pred_tagsets = list(pred_tagsets)
        if isinstance(true_tagsets, list):
            true_tagsets = list(true_tagsets)
            score += score_func(pred_tagsets, true_tagsets, labels)
    return score / len(srcids)

def get_micro_f1(true_mat, pred_mat):
    TP = np.sum(np.bitwise_and(true_mat==1, pred_mat==1))
    TN = np.sum(np.bitwise_and(true_mat==0, pred_mat==0))
    FN = np.sum(np.bitwise_and(true_mat==1, pred_mat==0))
    FP = np.sum(np.bitwise_and(true_mat==0, pred_mat==1))
    micro_prec = TP / (TP + FP)
    micro_rec = TP / (TP + FN)
    return 2 * micro_prec * micro_rec / (micro_prec + micro_rec)

def add_words_from_ts(learning_srcids, target_srcids):
    with open("Binarizer/mlb.pkl", 'rb') as f:
        mlb = pickle.load(f, encoding='bytes')
    with open("TS_Features/ebu3b_features.pkl", 'rb') as f:
        ts_features = pickle.load(f)
    ts2ir = TimeSeriesToIR(mlb=mlb)
    mlb_keys, Y_pred, Y_proba =\
            ts2ir.ts_to_ir(ts_features, learning_srcids, test_srcids)

def hamming_loss_func(pred_tagsets, true_tagsets, labels):
    incorrect_cnt = 0
    for tagset in pred_tagsets:
        if tagset not in true_tagsets:
            incorrect_cnt += 1
    for tagset in true_tagsets:
        if tagset not in pred_tagsets:
            incorrect_cnt += 1
    return incorrect_cnt / len(labels)

def accuracy_func(pred_tagsets, true_tagsets, labels=None):
    pred_tagsets = set(pred_tagsets)
    true_tagsets = set(true_tagsets)
    return len(pred_tagsets.intersection(true_tagsets))\
            / len(pred_tagsets.union(true_tagsets))

def get_accuracy(true_mat, pred_mat):
    acc_list = list()
    for true, pred in zip(true_mat, pred_mat):
        true_pos_indices = set(np.where(true==1)[0])
        pred_pos_indices = set(np.where(pred==1)[0])
        acc = len(pred_pos_indices.intersection(true_pos_indices)) /\
                len(pred_pos_indices.union(true_pos_indices))
        acc_list.append(acc)
    return np.mean(acc_list)

def hierarchy_accuracy_func(pred_tagsets, true_tagsets, labels=None):
    true_tagsets = deepcopy(true_tagsets)
    pred_tagsets = deepcopy(pred_tagsets)
    if not isinstance(pred_tagsets, list):
        pred_tagsets = list(pred_tagsets)
    union = 0
    intersection = 0
    for pred_tagset in deepcopy(pred_tagsets):
        if pred_tagset in true_tagsets:
            union += 1
            intersection += 1
            true_tagsets.remove(pred_tagset)
            pred_tagsets.remove(pred_tagset)
            continue
    depth_measurer = lambda x: tree_depth_dict[x]
    for pred_tagset in deepcopy(pred_tagsets):
        subclasses = subclass_dict[pred_tagset]
        lower_true_tagsets = [tagset for tagset in subclasses \
                              if tagset in true_tagsets]
        if len(lower_true_tagsets)>0:
            lower_true_tagsets = sorted(lower_true_tagsets,
                                        key=depth_measurer,
                                        reverse=False)
            lower_true_tagset = lower_true_tagsets[0]
            union += 1
            curr_score = tree_depth_dict[pred_tagset] /\
                            tree_depth_dict[lower_true_tagset]
            try:
                assert curr_score <= 1
            except:
                pdb.set_trace()
            intersection += curr_score
            pred_tagsets.remove(pred_tagset)
            true_tagsets.remove(lower_true_tagset)
    for pred_tagset in deepcopy(pred_tagsets):
        for true_tagset in deepcopy(true_tagsets):
            subclasses = subclass_dict[true_tagset]
            if pred_tagset in subclasses:
                union += 1
                curr_score = tree_depth_dict[true_tagset] /\
                                tree_depth_dict[pred_tagset]
                try:
                    assert curr_score <= 1
                except:
                    pdb.set_trace()

                intersection += curr_score
                pred_tagsets.remove(pred_tagset)
                true_tagsets.remove(true_tagset)
                break
    union += len(pred_tagsets) + len(true_tagsets)
    return intersection / union


def hierarchy_accuracy_func_deprecated(pred_tagsets, true_tagsets, labels):
    true_tagsets = deepcopy(true_tagsets)
    pred_tagsets = deepcopy(pred_tagsets)
    if not isinstance(pred_tagsets, list):
        pred_tagsets = list(pred_tagsets)
    union = 0
    intersection = 0
    for pred_tagset in deepcopy(pred_tagsets):
        if pred_tagset in true_tagsets:
            union += 1
            intersection += 1
            true_tagsets.remove(pred_tagset)
            pred_tagsets.remove(pred_tagset)
            continue
    for pred_tagset in deepcopy(pred_tagsets):
        subclasses = subclass_dict[pred_tagset]
        upper_true_tagsets = [tagset for tagset in subclasses \
                              if tagset in true_tagsets]
        if len(upper_true_tagsets)>0:
            upper_true_tagsets = sorted(upper_true_tagsets,
                                        key=tagset_lengther,
                                        reverse=True)
            upper_true_tagset = upper_true_tagsets[0]
            union += 1
            if upper_true_tagset=='hvac':
                curr_score = 0.5 #TODO: Fix these metric to use tree.
            else:
                curr_score = tagset_lengther(pred_tagset) \
                        / tagset_lengther(upper_true_tagset)
            try:
                assert curr_score < 1
            except:
                pdb.set_trace()
            intersection += curr_score
            pred_tagsets.remove(pred_tagset)
            true_tagsets.remove(upper_true_tagset)
    for pred_tagset in deepcopy(pred_tagsets):
        for true_tagset in deepcopy(true_tagsets):
            subclasses = subclass_dict[true_tagset]
            if pred_tagset in subclasses:
                union += 1
                curr_score = tagset_lengther(true_tagset) \
                        / tagset_lengther(pred_tagset)
                try:
                    assert curr_score < 1
                except:
                    pdb.set_trace()

                intersection += curr_score
                pred_tagsets.remove(pred_tagset)
                true_tagsets.remove(true_tagset)
                break
    union += len(pred_tagsets) + len(true_tagsets)
    return intersection / union



def micro_averaging(pred_Y, true_Y, labels):
    pass


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
        else:
            if tagset not in subclass_dict.keys():
                classname = tagset.split('_')[-1]
                subclass_dict[classname].append(tagset)
                subclass_dict[tagset] = []
                extend_tree(tagset_tree, classname, {tagset:[]})

def select_random_samples(building, \
                          srcids, \
                          n, \
                          use_cluster_flag,\
                          token_type='justseparate',
                          reverse=True,
                          cluster_dict=None,
                          shuffle_flag=True,
                         ):
    if not cluster_dict:
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
            sorted(cluster_dict.items(), key=length_counter, reverse=reverse))
        #n = len(sorted_cluster_dict) #TODO: Remove if not working well
        while len(sample_srcids) < n:
            cluster_dict_items = list(sorted_cluster_dict.items())
            if shuffle_flag:
                random.shuffle(cluster_dict_items)
            for cluster_num, srcid_list in cluster_dict_items:
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
                    use_brick_flag=False,
                    prev_step_data={
                        'learning_srcids':[],
                        'iter_cnt':0
                    },
                    oversample_flag=False
                   ):
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
    logging.info('{0}th Start learning CRF model'.format(\
                                                prev_step_data['iter_cnt']))


    ### TRAINING ###
    trainer = pycrfsuite.Trainer(verbose=False, algorithm='pa')
    # algorithm: {'lbfgs', 'l2sgd', 'ap', 'pa', 'arow'}
    trainer.set_params({'feature.possible_states': True,
                        'feature.possible_transitions': True})

    data_available_buildings = []
    learning_srcids = list()
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
        if prev_step_data['learning_srcids']:
            sample_srcid_list = [srcid for srcid in sentence_dict.keys() \
                                 if srcid in prev_step_data['learning_srcids']]
        else:
            sample_srcid_list = select_random_samples(building, \
                                                      label_dict.keys(), \
                                                      source_sample_num, \
                                                      use_cluster_flag)
        learning_srcids += sample_srcid_list
        
        if oversample_flag:
            sample_srcid_list = sample_srcid_list * \
                                floor(1000 / len(sample_srcid_list))

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
    if prev_step_data.get('learning_srcids_history'):
        assert set(prev_step_data['learning_srcids_history'][-1]) == set(learning_srcids)


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
        'source_building_count': len(building_list),
        'learning_srcids': sorted(learning_srcids)
    }
    store_model(model)
    os.remove(crf_model_file)

    logging.info("Finished!!!")

def crf_test(building_list,
             source_sample_num_list,
             target_building,
             token_type='justseparate',
             label_type='label',
             use_cluster_flag=False,
             use_brick_flag=False,
             learning_srcids=[]
            ):
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
                                    'this.source_list.{0}.length=={1}'.\
                                    format(building, source_sample_num)})
        result_metadata['source_cnt_list'].append([building, source_sample_num])
    model_query['$and'].append(model_metadata)
    model_query['$and'].append({'source_building_count':len(building_list)})
    if learning_srcids:
        model_query = {'learning_srcids': sorted(learning_srcids)}
    try:
        model = get_model(model_query)
    except:
        pdb.set_trace()
    result_metadata['source_list'] = model['source_list']

    if not learning_srcids:
        learning_srcids = sorted(list(reduce(adder, model['source_list'].values())))
    
    result_metadata['learning_srcids'] = learning_srcids

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
    #plt.scatter(score_list, error_rate_list, alpha=0.3)
    #plt.plot(score_list, p(score_list), "r--")
    #save_fig(plt.gcf(), error_plot_file)
    step_data = {
        'learning_srcids': learning_srcids,
        'result': resulter.get_summary()
    }
    return step_data


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

def tagset_lengther(tagset):
    return len(tagset.split('_'))

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

def tagsets_prediction(classifier, vectorizer, binarizer, \
                           phrase_dict, srcids, source_target_buildings, \
                       eda_flag, point_classifier, ts2ir=None):
    global point_tagsets
    logging.info('Start prediction')
    if ts2ir:
        phrase_dict = augment_ts(phrase_dict, srcids, ts2ir)
    doc = [' '.join(phrase_dict[srcid]) for srcid in srcids]

    manual_filter_flag = False
    if manual_filter_flag:
        for srcid in srcids:
            phrase_dict[srcid] = filt(phrase_dict[srcid])


    if eda_flag:
        vect_doc = eda_vectorizer(vectorizer, doc, \
                                       source_target_buildings, srcids)
    else:
        vect_doc = vectorizer.transform(doc) # should this be fit_transform?

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
    #point_mat = point_classifier.predict(vect_doc)
    logging.info('Finished point prediction')
    #prob_mat = classifier.predict_proba(vect_doc)
    pred_tagsets_dict = dict()
    pred_certainty_dict = dict()
    pred_point_dict = dict()
    for i, (srcid, pred) in enumerate(zip(srcids, pred_mat)):
    #for i, (srcid, pred, point_pred) \
            #in enumerate(zip(srcids, pred_mat, point_mat)):
        pred_tagsets_dict[srcid] = binarizer.inverse_transform(\
                                        np.asarray([pred]))[0]
        #pred_tagsets_dict[srcid] = list(binarizer.inverse_transform(pred)[0])
        #pred_point_dict[srcid] = point_tagsets[point_pred]
        #pred_vec = [prob[i][0] for prob in prob_mat]
        #pred_certainty_dict[srcid] = pred_vec
        pred_certainty_dict[srcid] = 0
    pred_certainty_dict = OrderedDict(sorted(pred_certainty_dict.items(), \
                                             key=itemgetter(1), reverse=True))
    logging.info('Finished prediction')
    return pred_tagsets_dict, pred_certainty_dict, pred_point_dict

def tagsets_evaluation(truths_dict, pred_tagsets_dict, pred_certainty_dict,\
                       srcids, pred_point_dict, phrase_dict, debug_flag=False, classifier=None, vectorizer=None):
    result_dict = defaultdict(dict) # TODO: In case of a bug, it was defaultdict
    sorted_result_dict = OrderedDict()
    incorrect_tagsets_dict = dict()
    correct_cnt = 0
    incorrect_cnt = 0
    point_correct_cnt = 0
    point_incorrect_cnt = 0
    empty_point_cnt = 0
    unknown_reason_cnt = 0
    undiscovered_point_cnt = 0
    invalid_point_cnt = 0
    manual_filter_flag = False
    if manual_filter_flag:
        for srcid in srcids:
            truths_dict[srcid] = filt(truths_dict[srcid])
    disch2sup = lambda s: s.replace('discharge', 'supply')
#    for srcid, pred_tagsets in pred_tagsets_dict.items():
    truths_dict = dict([(srcid, list(map(disch2sup, tagsets))) \
                         for srcid, tagsets in truths_dict.items()])
    accuracy = get_score(pred_tagsets_dict, truths_dict, srcids, accuracy_func,
                         tagset_list)
    hierarchy_accuracy = get_score(pred_tagsets_dict, truths_dict, srcids,
                                   hierarchy_accuracy_func, tagset_list)
    hamming_loss = get_score(pred_tagsets_dict, truths_dict, srcids,
                             hamming_loss_func, tagset_list)
    subset_accuracy = get_score(pred_tagsets_dict, truths_dict, srcids,
                                subset_accuracy_func, tagset_list)

    pred_tagsets_list = [pred_tagsets_dict[srcid] for srcid in srcids]
    true_tagsets_list = [list(map(disch2sup, truths_dict[srcid])) for srcid in srcids]
    eval_binarizer = MultiLabelBinarizer().fit(pred_tagsets_list +
                                               true_tagsets_list)
    _, _, macro_f1, _ = precision_recall_fscore_support(\
                            eval_binarizer.transform(true_tagsets_list),\
                            eval_binarizer.transform(pred_tagsets_list),\
                            average='macro')
    _, _, weighted_f1, _ = precision_recall_fscore_support(\
                            eval_binarizer.transform(true_tagsets_list),\
                            eval_binarizer.transform(pred_tagsets_list),\
                            average='weighted')
    true_mat = eval_binarizer.transform(true_tagsets_list)
    pred_mat = eval_binarizer.transform(pred_tagsets_list)
    f1_list = []
    prec_list = []
    rec_list = []
    sup_list = []
    for i in range(0, true_mat.shape[1]):
        true = true_mat[:,i]
        pred = pred_mat[:,i]
        if np.sum(true)!=0 or np.sum(pred)!=0:
            prec, rec, f1, support = precision_recall_fscore_support(true, pred)
            prec_list.append(prec)
            rec_list.append(rec)
            f1_list.append(f1)
            sup_list.append(support)
    #macro_f1 = np.mean(f1_list)
    #weighted_f1 = np.mean(f1_list)
    print('avg prec: {0}'.format(np.mean(prec_list)))
    print('avg rec: {0}'.format(np.mean(rec_list)))
    print('avg f1: {0}'.format(np.mean(f1_list)))
    micro_f1 = get_micro_f1(true_mat, pred_mat)

    for srcid in srcids:
        pred_tagsets = pred_tagsets_dict[srcid]
        true_tagsets = truths_dict[srcid]
        one_result = {
            'tagsets': pred_tagsets,
            'certainty': pred_certainty_dict[srcid]
        }
        need_review = False
        true_point = None
        for tagset in true_tagsets:
            if tagset in point_tagsets:
                true_point = tagset
                break
        result_dict['A'][srcid] = accuracy_func(pred_tagsets, true_tagsets)
        result_dict['HA'][srcid] = hierarchy_accuracy_func(pred_tagsets,
                                                           true_tagsets)
        result_dict['phrase_usage'][srcid] = determine_used_phrases(
                                                phrase_dict[srcid],
                                                pred_tagsets)
        if set(true_tagsets) == set(pred_tagsets):
            correct_cnt += 1
            one_result['correct?'] = True
            result_dict['correct'][srcid] = one_result
            #result_dict['correct'][srcid] = pred_tagsets
            point_correct_cnt += 1
            need_review = True
        else:
            incorrect_cnt += 1
            one_result['correct?'] = False
            result_dict['incorrect'][srcid] = one_result
            if not true_point:
                invalid_point_cnt += 1
                continue
            found_point = None
            #found_point = pred_point_dict[srcid]
            for tagset in pred_tagsets:
                if tagset in point_tagsets:
                    found_point = tagset
                    break
            if found_point in ['none', None]:
                empty_point_cnt += 1
                need_review = True
            elif found_point != true_point and \
                 found_point.replace('supply', 'discharge') != true_point and\
                 found_point.replace('discharge', 'supply') != true_point:
            #elif found_point != true_point:
                point_incorrect_cnt += 1
                print("INCORRECT POINT ({0}) FOUND: {1} -> {2}"\
                      .format(srcid, true_point, found_point))
                need_review = True
            else:
                unknown_reason_cnt += 1
                point_correct_cnt += 1
                need_review = True

#True        if srcid == '513_0_3006645':

        if need_review and debug_flag:
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
    print('invalid points: ', invalid_point_cnt)
    print('-----------')
    result_dict['point_precision'] = point_precision
    result_dict['precision'] = precision
    result_dict['point_recall'] = point_recall
    result_dict['point_correct_cnt'] = point_correct_cnt
    result_dict['point_incorrect_cnt'] = point_incorrect_cnt
    result_dict['unfound_point_cnt'] = empty_point_cnt
    result_dict['hamming_loss'] = hamming_loss
    result_dict['accuracy'] = accuracy
    result_dict['hierarchy_accuracy'] = hierarchy_accuracy
    result_dict['subset_accuracy'] = subset_accuracy
    result_dict['macro_f1'] = macro_f1
    result_dict['micro_f1'] = micro_f1
    result_dict['weighted_f1'] = weighted_f1
    pp.pprint(dict([(k,v) for k, v in result_dict.items() \
                    if k not in ['correct', 'incorrect', 'phrase_usage', 'HA', 'A']]))
    return dict(result_dict)

def tagsets_evaluation_deprecated(truths_dict, pred_tagsets_dict, \
                                  pred_certainty_dict, srcids, pred_point_dict):
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

class ProjectClassifier():
    def __init__(self, base_classifier, binarizer, vectorizer, subclass_dict,
                n_jobs=1):
        self.binarizer = binarizer
        self.vectorizer = vectorizer
        self.base_classifier = base_classifier
        self.make_proj_vec()
        self.subclass_dict = subclass_dict
        self.n_jobs = n_jobs

        self.lower_y_index_list = list()
        self.upper_y_index_list = list()
        for i, tagset in enumerate(self.binarizer.classes_):
            found_upper_tagsets = find_keys(tagset, self.subclass_dict, check_in)
            upper_tagsets = [ts for ts in self.binarizer.classes_ \
                             if ts in found_upper_tagsets]
            try:
                assert len(found_upper_tagsets) == len(upper_tagsets)
            except:
                pdb.set_trace()
            self.upper_y_index_list.append([
                np.where(self.binarizer.classes_ == ts)[0][0]
                                            for ts in upper_tagsets])
            lower_y_indices = list()
            subclasses = self.subclass_dict.get(tagset)
            if not subclasses:
                subclasses = []
            for ts in subclasses:
                indices = np.where(self.binarizer.classes_ == ts)[0]
                if len(indices)>1:
                    assert False
                elif len(indices==1):
                    lower_y_indices.append(indices[0])
            self.lower_y_index_list.append(lower_y_indices)

    def make_proj_vec(self):
        #vec = np.zeros((len(self.vectorizer.vocabulary), self.binarizer.classes_))
        vec_list = list()
        for tagset in self.binarizer.classes_:
            tags = tagset.replace('_', ' ')
            vectorized_tags = np.array([1 if v > 0 else 0 for v in
                               self.vectorizer.transform([tags]).toarray()[0]])
            vec_list.append(vectorized_tags)
        self.proj_vectors = np.vstack(vec_list)

    def _proj_x(self, X, proj_vector):
        #return np.hstack([X[:, j] for j, v in enumerate(proj_vector) \
        #                  if v == 1])
        return X[:, [i for i, v in enumerate(proj_vector) if v == 1]]

    def _X_formatting(self, X):
        if issparse(X):
            return X.todense()
        elif not isinstance(X, np.matrix):
            return np.asmatrix(X)
        else:
            return X

    def serial_fit(self, X, Y):
        X = self._X_formatting(X)
        self.base_classifiers = list()
        for i, (y, proj_vector) in enumerate(zip(Y.T, self.proj_vectors)):
            #proj_X = np.hstack([X[:,i] for i, v in enumerate(proj_vector) if v==1])
            proj_X = self._proj_x(X, proj_vector)
            self.base_classifiers.append(deepcopy(self.base_classifier)\
                                        .fit(proj_X, y))
    def fit(self, X, Y):
        if self.n_jobs == 1:
            return self.serial_fit(X, Y)
        else:
            return self.parallel_fit(X, Y)

    def sub_fit(self, X, Y, i):
        if i%200==0:
            logging.info('{0}th learning step'.format(i))
        proj_X = self._proj_x(X, proj_vector)
        #self.base_classifiers.append(deepcopy(self.base_classifier)\
        #                            .fit(proj_X, y))
        return deepcopy(self.base_classifier).fit(proj_X, y)

    def parallel_fit(self, X, Y):
        p = Pool(self.n_jobs)
        mapped_sub_fit = partial(self.sub_fit, X, Y)
        self.base_classifiers = p.map(mapped_sub_fit, range(0,Y.shape[1]))
        p.close()

    def predict(self, X):
        Ys = list()
        X = self._X_formatting(X)
        for proj_vector, base_classifier \
                in zip(self.proj_vectors, self.base_classifiers):
#            proj_X = np.hstack([X[:, i] for i, v in enumerate(proj_vector) if v==1])
            proj_X = self._proj_x(X, proj_vector)
            Ys.append(base_classifier.predict(proj_X))
        Y = np.vstack(Ys).T
        return self._distill_Y(Y)

    def _distill_Y(self, Y):
        logging.info('Start distilling')
        """
        discharge_supply_map = dict()
        for i, tagset in enumerate(self.binarizer.classes_):
            if 'discharge' in tagset:
                discharge_supply_map[i] = np.where(binarizer.classes_ == \
                    tagset.replace('discharge', 'supply'))[0]
        for i_discharge, i_supply in discharge_supply_map.items():
            pdb.set_trace()
            discharge_indices = np.where(Y[:, i_discharge] == 1)
            Y[discharge_indices, i_discharge] = 0
            Y[discharge_indices, i_supply] = 1
        """
        new_Y = deepcopy(Y)
        for i, y in enumerate(Y):
            new_y = np.zeros(len(y))
            for j, one_y in enumerate(y):
                subclass_y_indices = self.lower_y_index_list[j]
                if 1 in y[subclass_y_indices]:
                    new_y[j] = 0
                else:
                    new_y[j] = one_y
            new_Y[i,:] = new_y
        logging.info('Finished distilling')
        return new_Y

    def augment_biased_sample(self, X, y):
        rnd_sample_num = int(X.shape[0] * 0.05)
        sub_X = X[np.where(y==1)]
        added_X_list = list()
        if sub_X.shape[0] == 0:
            return X, y
#        added_X = list()
        for i in range(0, rnd_sample_num):
            if self.use_brick_flag:
                sub_brick_indices = np.intersect1d(np.where(y==1)[0],
                                               self.brick_indices)
                if len(sub_brick_indices)==0:
                    x_1 = sub_X[random.randint(0, sub_X.shape[0] - 1)]
                else:
                    sub_brick_X = X[sub_brick_indices]
                    x_1 = sub_brick_X[random.randint(0, sub_brick_X.shape[0]-1)]
            else:
                x_1 = sub_X[random.randint(0, sub_X.shape[0] - 1)]
            x_2 = sub_X[random.randint(0, sub_X.shape[0] - 1)]
            avg_factor = random.random()
            #new_x = x_1 + (x_2 - x_1) * avg_factor
            if i % 2 == 0:
                new_x = x_1
            else:
                new_x = x_2
            y = np.append(y, 1)
            added_X_list.append(new_x)
        X = np.vstack([X] + added_X_list)
        assert X.shape[0] == y.shape[0]
        return X, y

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

def augment_true_mat_with_superclasses_deprecated_and_incorrect(binarizer, given_label_vect,\
                                       subclass_dict=subclass_dict):
    tagsets = binarizer.inverse_transform([given_label_vect])
    updated_tagsets = [find_keys(tagset, subclass_dict, check_in)\
                       for tagset in tagsets]
    pdb.set_trace()
    return binarizer.transform([list(set(tagsets + updated_tagsets))])

class SingleProjectClassifier():

    def __init__(self, base_classifier, mask):
        self.base_classifier = base_classifier
        self.base_mask = mask
        """
        for i, tagset in enumerate(self.binarizer.classes_):
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
        #self.mask = np.concatenate([self.base_mask, extended_feature_dims])
        mask = [i for i, v in enumerate(self.base_mask) if v == 1] +\
                range(self.base_mask, X.shape[1])
        X = X[:, mask]
        #X = X[:, np.where(self.mask==1)[0]]
        self.base_classifier.fit(X, y)

    def predict(self, X):
        assert len(self.mask) == X.shape[1]
        X = X[:, np.where(self.mask==1)[0]]
        return self.base_classifier.predict(X)


class VotingClassifier():
    def __init__(self, binarizer, vectorizer, tagset_tree, tagset_list):
        self.binarizer = binarizer
        self.tagset_tree = tagset_tree
        self.vectorizer = vectorizer
        self.tagset_list = tagset_list

    def fit(self, X, Y):
        pass

    def tagset_score(self, tags, tagset):
        if set(tags.tolist()) == set(['max', 'supply', 'air', 'static',
                                   'pressure', 'setpoint'])\
           and tagset == 'max_load_setpoint':
            #pdb.set_trace()
            pass
        tagset_tags = tagset.split('_')
        tagset_slots = dict([(tag, 0) for tag in tagset_tags])
        for tag in tags:
            if tag in tagset_tags:
                tagset_slots[tag] += 1
        used_tags = sum(tagset_slots.values())
        even_dist_num = used_tags / len(tagset_tags)
        max_score = len(tagset_tags) * np.log(even_dist_num+1)
        log_scorer = lambda x: np.log(x+1)
        curr_score = sum(map(log_scorer, tagset_slots.values()))
        return curr_score / max_score

    def predict_tagset(self, tags):
        pred_tagsets = list()
        for tagset in self.tagset_list:
            if self.tagset_score(tags, tagset) >= 0.92:
                pred_tagsets.append(tagset)
        return pred_tagsets

    def predict(self, X):
        phrases_mat = self.vectorizer.inverse_transform(X)
        pred_tagsets = list()
        for phrases in phrases_mat:
            pred_tagsets.append(self.predict_tagset(phrases))

        return pred_tagsets


class StructuredClassifierChain():

    def __init__(self, base_classifier, binarizer, subclass_dict,
                 vocabulary_dict, n_jobs=1, use_brick_flag=False, vectorizer=None):
        self.vectorizer = vectorizer
        self.prob_flag = False
        self.use_brick_flag = use_brick_flag
        self.n_jobs = n_jobs
        self.vocabulary_dict = vocabulary_dict
        self.subclass_dict = subclass_dict
        self.base_classifier = base_classifier
        self.binarizer = binarizer
        self.upper_y_index_list = list()
        self.lower_y_index_list = list()
        self.base_classifiers = list()
        for i, tagset in enumerate(self.binarizer.classes_):
            found_upper_tagsets = find_keys(tagset, self.subclass_dict, check_in)
            upper_tagsets = [ts for ts in self.binarizer.classes_ \
                             if ts in found_upper_tagsets]
            try:
                assert len(found_upper_tagsets) == len(upper_tagsets)
            except:
                #pdb.set_trace()
                pass
            self.upper_y_index_list.append([
                np.where(self.binarizer.classes_ == ts)[0][0]
                                            for ts in upper_tagsets])
            lower_y_indices = list()
            subclasses = self.subclass_dict.get(tagset)
            if not subclasses:
                subclasses = []
            for ts in subclasses:
                indices = np.where(self.binarizer.classes_ == ts)[0]
                if len(indices)>1:
                    assert False
                elif len(indices==1):
                    lower_y_indices.append(indices[0])
            self.lower_y_index_list.append(lower_y_indices)
            self.base_classifiers.append(deepcopy(self.base_classifier))
        #self.make_proj_vec()
        self.vectorizer = None

    def make_proj_vec(self):
        vec_list = list()
        for tagset in self.binarizer.classes_:
            tags = tagset.replace('_', ' ')
            vectorized_tags = np.array([1 if v > 0 else 0 for v in
                               self.vectorizer.transform([tags]).toarray()[0]])
            vec_list.append(vectorized_tags)
        self.proj_vectors = np.vstack(vec_list)

    def _augment_X(self, X, Y):
        return np.hstack([X, Y*2])

    def _find_brick_indices(self, X, Y, orig_sample_num):
        brick_indices = list()
        for i, y in enumerate(Y):
            if i >= orig_sample_num and np.sum(y) == 1: #TODO: Need to fix orig_sample_num to consider negative samples
                brick_indices.append(i)
        return np.array(brick_indices)
        #return np.where(np.array(list(map(np.sum, Y))) == 1 )[0]


    def fit(self, X, Y, orig_sample_num = 0):
        if self.use_brick_flag:
            self.brick_indices = self._find_brick_indices(X, Y, \
                                                          orig_sample_num)
        if self.n_jobs == 1:
            return self.serial_fit(X, Y)
        else:
            return self.parallel_fit(X, Y)

    def serial_fit(self, X, Y):
        logging.info('Start fitting')
        X = self.conv_array(X)
        Y = self._augment_labels_superclasses(Y)
        for i, y in enumerate(Y.T):
            """
            sub_Y = Y[:, self.upper_y_index_list[i]]
            augmented_X = self._augment_X(X, sub_Y)
            unbiased_X, unbiased_y = self.augment_biased_sample(augmented_X, y)
            base_classifier = deepcopy(base_classifier)
            tagset = self.binarizer.classes_[i]
            tags = tagset.split('_')
            base_classifier.steps[0] = ('feature_selection', SelectKBest(chi2, k=len(tags)+3))
            try:
                self.base_classifiers[i].fit(unbiased_X, unbiased_y)
            #self.base_classifiers[i].fit(augmented_X, y)
            except:
                pass
            """
            self.base_classifiers[i] = self.sub_fit(X, Y, i)

        logging.info('Finished fitting')

    def parallel_fit(self, X,Y):
        p = Pool(self.n_jobs)
        Y = self._augment_labels_superclasses(Y)
        mapped_sub_fit = partial(self.sub_fit, X, Y)
        self.base_classifiers = p.map(mapped_sub_fit, range(0,Y.shape[1]))
        p.close()

    def augment_biased_sample(self, X, y):
        rnd_sample_num = int(X.shape[0] * 0.05)
        sub_X = X[np.where(y==1)]
        added_X_list = list()
        if sub_X.shape[0] == 0:
            return X, y
#        added_X = list()
        for i in range(0, rnd_sample_num):
            if self.use_brick_flag:
                sub_brick_indices = np.intersect1d(np.where(y==1)[0],
                                               self.brick_indices)
                if len(sub_brick_indices)==0:
                    x_1 = sub_X[random.randint(0, sub_X.shape[0] - 1)]
                else:
                    sub_brick_X = X[sub_brick_indices]
                    x_1 = sub_brick_X[random.randint(0, sub_brick_X.shape[0]-1)]
            else:
                x_1 = sub_X[random.randint(0, sub_X.shape[0] - 1)]
            x_2 = sub_X[random.randint(0, sub_X.shape[0] - 1)]
            avg_factor = random.random()
            #new_x = x_1 + (x_2 - x_1) * avg_factor
            if i % 2 == 0:
                new_x = x_1
            else:
                new_x = x_2
            y = np.append(y, 1)
            added_X_list.append(new_x)
        X = np.vstack([X] + added_X_list)
        assert X.shape[0] == y.shape[0]
        return X, y

    def sub_fit(self, X, Y, i):
        if i%200==0:
            logging.info('{0}th learning step'.format(i))
        if i==857:
            transformer = lambda x: [vocab for vocab, v \
                in self.vocabulary_dict.items() if x[0,v]>0]
            #pdb.set_trace()

        y = Y.T[i]
        sub_Y = Y[:, self.upper_y_index_list[i]]
        augmented_X = self._augment_X(X, sub_Y)
        unbiased_X, unbiased_y = self.augment_biased_sample(augmented_X, y)
        base_classifier = deepcopy(self.base_classifier)
        #tagset = self.binarizer.classes_[i]
        #tags = tagset.split('_')
        #base_classifier.steps[0] = ('feature_selection', SelectKBest(chi2, k=len(tags) + 3))
        try:
            base_classifier.fit(unbiased_X, unbiased_y)
            #base_classifier.fit(augmented_X, y)
        except:
            pass
        return base_classifier

    def sub_fit_proj(self, X, Y, i):
        base_base_classifier = deepcopy(self.base_classifier)
        y = Y.T[i]
        tags = self.binarizer.classes_[i].split('_')
        mask = self.proj_vectors
        base_classifier = SinglProjectClassifier(base_base_classifier, mask)
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
        Y = np.zeros((X.shape[0], len(self.binarizer.classes_)))
        for i, (upper_y_indices, base_classifier) \
                in enumerate(zip(self.upper_y_index_list,
                                 self.base_classifiers)):
            try:
                assert sum([i <= y_index for y_index in upper_y_indices]) == 0
            except:
                #pdb.set_trace()
                [y_index for y_index in upper_y_indices if y_index < i]
            sub_Y = Y[:, upper_y_indices]
            augmented_X = self._augment_X(X, sub_Y)
            if i==414 and X.shape[0]>800 and False:
                filt = base_classifier.steps[0][1]
                filtered = filt.inverse_transform(filt.transform([augmented_X[i]]))
                #print('FILTERED: ', [vocab for vocab, j in self.vocabulary_dict.items() if filtered[0][j]>0])
                #print('FROM:', [vocab for vocab, j in self.vocabulary_dict.items() if augmented_X[i][j] > 0 ])
                transformer = lambda x: [vocab for vocab, v \
                                         in self.vocabulary_dict.items() if x[v]>0]
                #pdb.set_trace() # Check why supply fan is not deteced
            try:
                if self.prob_flag:
                    prob_y = base_classifier.predict_proba(augmented_X)
                    pred_y = np.array([prob[1] for prob in prob_y])
                else:
                    pred_y = base_classifier.predict(augmented_X)
            except:
                pred_y = np.zeros(augmented_X.shape[0])
            Y[:, i] = pred_y
        Y = self._distill_Y(Y)
        logging.info('Finished predicting')
        return Y

    def _distill_Y(self, Y):
        logging.info('Start distilling')
        # change discharge to supply at the labels 
        # (not in the prediction but in the results)
        discharge_supply_map = dict()
        for i, tagset in enumerate(self.binarizer.classes_):
            if 'discharge' in tagset:
                discharge_supply_map[i] = np.where(self.binarizer.classes_ == \
                    tagset.replace('discharge', 'supply'))[0]
        for i_discharge, i_supply in discharge_supply_map.items():
            discharge_indices = np.where(Y[:, i_discharge] == 1)
            Y[discharge_indices, i_discharge] = 0
            try:
                Y[discharge_indices, i_supply] = 1
            except:
                pdb.set_trace()

        if self.prob_flag:
            new_Y = np.zeros(Y.shape)
            for i, y in enumerate(Y):
                new_Y[i] = np.array([1 if prob>0.5 else 0 for prob in y])
            Y = new_Y

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
            tagsets = self.binarizer.inverse_transform(vect)[0]
            updated_tagsets = reduce(adder, [
                                find_keys(tagset, self.subclass_dict, check_in)
                                for tagset in tagsets], [])
            #TODO: This is bad code. need to be fixed later.
            finished = False
            while not finished:
                try:
                    new_row = self.binarizer.transform([list(set(list(tagsets)
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

    def __init__(self, base_classifier, binarizer, \
                 subclass_dict=subclass_dict, tagset_tree=tagset_tree):
        self.subclass_dict = subclass_dict
        self.base_classifier = base_classifier
        self.binarizer = binarizer
        self.tagset_tree = tagset_tree
        self.classifier_chain = self._init_classifier_chain(self.tagset_tree)
        self.index_tagset_dict = dict([(tagset, i) for i, tagset \
                                       in enumerate(self.binarizer.classes_)])

    def _augment_labels_superclasses(self, Y):
        logging.info('Start augmenting label mat with superclasses')
        Y = lil_matrix(Y)
        for i, vect in enumerate(Y):
            tagsets = self.binarizer.inverse_transform(vect)[0]
            updated_tagsets = reduce(adder, [
                                find_keys(tagset, self.subclass_dict, check_in)
                                for tagset in tagsets], [])
            #TODO: This is bad code. need to be fixed later.
            finished = False
            while not finished:
                try:
                    new_row = self.binarizer.transform([list(set(list(tagsets)
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
        # Y and self.binarizer should be synchronized initially.
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
            self.pred_Y = np.zeros((X.shape[0], len(self.binarizer.classes_)))
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

def augment_ts(phrase_dict, srcids, ts2ir):
    with open(ts_feature_filename, 'rb') as fp:
        ts_features = pickle.load(fp, encoding='bytes')
    ts_tags_pred = ts2ir.predict(ts_features, srcids)

    tag_binarizer = ts2ir.get_binarizer()
    pred_tags_list = tag_binarizer.inverse_transform(ts_tags_pred)

    for srcid, pred_tags in zip(srcids, pred_tags_list):
        phrase_dict[srcid] += list(pred_tags)
    return phrase_dict


def tree_flatter(tree, init_flag=True):
    branches_list = list(tree.values())
    d_list = list(tree.keys())
    for branches in branches_list:
        for branch in branches:
            added_d_list = tree_flatter(branch)
            d_list = [d for d in d_list if d not in added_d_list]\
                    + added_d_list
    return d_list

def filt(tagsets):
    return [tagset for tagset in tagsets if tagset.split('-')[0] not in \
            ['building', 'networkadapter', \
             'rightidentifier', 'leftidentifier']]


def parameter_validation(vect_doc, truth_mat, srcids, params_list_dict,\
                         meta_classifier, vectorizer, binarizer, \
                         source_target_buildings, eda_flag):
    #best_params = {'n_estimators': 50, 'criterion': 'entropy', 'max_features': 'auto', 'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 2}
    #best_params = {'criterion': 'entropy'}
    #best_params = {'loss': 'exponential', 'learning_rate': 0.01, 'criterion': 'friedman_mse', 'max_features': None, 'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 2}

        #tagset_classifier = RandomForestClassifier(n_estimators=100,
        #                                           random_state=0,\
        #                                           n_jobs=n_jobs)
    best_params = {'learning_rate':0.1, 'subsample':0.25}
    #best_params = {'C':0.4, 'solver': 'liblinear'}
    return meta_classifier(**best_params) # Pre defined setup.
    #best_params = {'n_estimators': 120, 'n_jobs':7}
    #return meta_classifier(**best_params)

    token_type = 'justseparate'
    results_dict = dict()
    for key, values in params_list_dict.items():
        results_dict[key] = {'ha': [0]*len(values),
                             'a': [0]*len(values),
                             'mf1': [0]*len(values)}
    avg_num = 3
    for i in range(0,avg_num):
        learning_indices = random.sample(range(0, len(srcids)),
                                         int(len(srcids)/2))
        validation_indices = [i for i in range(0, len(srcids))
                              if i not in learning_indices]
        learning_srcids = [srcids[i] for i
                                    in learning_indices]
        validation_srcids = [srcids[i] for i
                             in validation_indices]
        for key, values in params_list_dict.items():
            for j, value in enumerate(values):
                params = {key: value}
                classifier = meta_classifier(**params)
                classifier.fit(vect_doc[learning_indices], \
                               truth_mat[learning_indices].toarray())

                validation_sentence_dict, \
                validation_token_label_dict, \
                validation_truths_dict, \
                validation_phrase_dict = get_multi_buildings_data(\
                                            source_target_buildings, validation_srcids, \
                                            eda_flag, token_type)

                validation_pred_tagsets_dict, \
                validation_pred_certainty_dict, \
                _ = tagsets_prediction(classifier, vectorizer, binarizer, \
                                   validation_phrase_dict, validation_srcids, \
                                   source_target_buildings, eda_flag, None,
                                       ts2ir=None)
                validation_result = tagsets_evaluation(validation_truths_dict, \
                                                       validation_pred_tagsets_dict, \
                                                       validation_pred_certainty_dict,\
                                                       validation_srcids, \
                                                       None, \
                                                       validation_phrase_dict, \
                                                       debug_flag=False,
                                                       classifier=classifier, \
                                                       vectorizer=vectorizer)
                results_dict[key]['ha'][j] += validation_result['hierarchy_accuracy']
                results_dict[key]['a'][j] += validation_result['accuracy']
                results_dict[key]['mf1'][j] += validation_result['macro_f1']
                results_dict[key]['macro_f1'][j] += validation_result['macro_f1']
    best_params = dict()
    for key, results in results_dict.items():
        metrics = results_dict[key]['mf1']
        best_params[key] = params_list_dict[key][metrics.index(max(metrics))]
    classifier = meta_classifier(**best_params)
    classifier.fit(vect_doc[learning_indices], \
                   truth_mat[learning_indices].toarray())

    validation_sentence_dict, \
    validation_token_label_dict, \
    validation_truths_dict, \
    validation_phrase_dict = get_multi_buildings_data(\
                                source_target_buildings, validation_srcids, \
                                eda_flag, token_type)

    validation_pred_tagsets_dict, \
    validation_pred_certainty_dict, \
    _ = tagsets_prediction(classifier, vectorizer, binarizer, \
                       validation_phrase_dict, validation_srcids, \
                       source_target_buildings, eda_flag, None,
                           ts2ir=None)
    validation_result = tagsets_evaluation(validation_truths_dict, \
                                           validation_pred_tagsets_dict, \
                                           validation_pred_certainty_dict,\
                                           validation_srcids, \
                                           None, \
                                           validation_phrase_dict, \
                                           debug_flag=False,
                                           classifier=classifier, \
                                           vectorizer=vectorizer)
    best_ha = validation_result['hierarchy_accuracy']
    best_a = validation_result['accuracy']
    best_mf1 = validation_result['macro_f1']

    return meta_classifier(**best_params)



def build_tagset_classifier(building_list, target_building,\
                            test_sentence_dict, test_token_label_dict,\
                            learning_phrase_dict, test_phrase_dict,\
                            learning_truths_dict,\
                            learning_srcids, test_srcids,\
                            tagset_list, eda_flag, use_brick_flag,
                            source_target_buildings,
                            n_jobs,
                            ts_flag,
                            negative_flag,
                            validation_truths_dict={}
                           ):
    validation_srcids = list(validation_truths_dict.keys())
    learning_srcids = deepcopy(learning_srcids)
    orig_learning_srcids = deepcopy(learning_srcids)
    global total_srcid_dict
    global point_tagsets
    global tagset_classifier_type
    global tree_depth_dict
#    source_target_buildings = list(set(building_list + [target_building]))

    orig_sample_num = len(learning_srcids)
    new_tagset_list = tree_flatter(tagset_tree, [])
    new_tagset_list = new_tagset_list + [ts for ts in tagset_list \
                                     if ts not in new_tagset_list]
    tagset_list = new_tagset_list

    tagset_binarizer = MultiLabelBinarizer(tagset_list)
    tagset_binarizer.fit([tagset_list])
    assert tagset_list == tagset_binarizer.classes_.tolist()

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

    # Define Vectorizer
    tokenizer = lambda x: x.split()
    tagset_vectorizer = TfidfVectorizer(tokenizer=tokenizer,\
                                        vocabulary=vocab_dict)
    #tagset_vectorizer = CountVectorizer(tokenizer=tokenizer,\
    #                                    vocabulary=vocab_dict)

    learning_point_dict = dict()
    for srcid, tagsets in chain(learning_truths_dict.items(),
                                validation_truths_dict.items()):
        point_tagset = 'none'
        for tagset in tagsets:
            if tagset in point_tagsets:
                point_tagset = tagset
                break
        learning_point_dict[srcid] = point_tagset
    learning_point_dict['dummy'] = 'unknown'

    ts2ir = None
    ts_learning_srcids = list()
    if ts_flag:
        learning_tags_dict = dict([(srcid, splitter(tagset)) for srcid, tagset
                                   in learning_point_dict.items()])
        tag_binarizer = MultiLabelBinarizer()
        tag_binarizer.fit(map(splitter, learning_point_dict.values()))
        with open(ts_feature_filename, 'rb') as fp:
            ts_features = pickle.load(fp, encoding='bytes')
        new_ts_features = list()
        for ts_feature in ts_features:
            feats = ts_feature[0]
            srcid = ts_feature[2]
            if srcid in learning_srcids + validation_srcids:
                point_tagset = learning_point_dict[srcid]
                point_tags = point_tagset.split('_')
                point_vec = tag_binarizer.transform([point_tags])
                new_feature = [feats, point_vec, srcid]
                new_ts_features.append(new_feature)
            elif srcid in test_srcids:
                new_ts_features.append(ts_feature)
        ts_features = new_ts_features

        ts2ir = TimeSeriesToIR(mlb=tag_binarizer)
        ts2ir.fit(ts_features, learning_srcids, validation_srcids, learning_tags_dict)
        learning_ts_tags_pred = ts2ir.predict(ts_features, learning_srcids)
        for srcid, ts_tags in zip(learning_srcids, \
                                  tag_binarizer.inverse_transform(
                                      learning_ts_tags_pred)):
            #learning_phrase_dict[srcid] += list(ts_tags)
            ts_srcid = srcid + '_ts'
            learning_phrase_dict[ts_srcid] = learning_phrase_dict[srcid]\
                                                + list(ts_tags)
            ts_learning_srcids.append(ts_srcid)
            learning_truths_dict[ts_srcid] = learning_truths_dict[srcid]

        test_ts_tags_pred = ts2ir.predict(ts_features, test_srcids)
        for srcid, ts_tags in zip(test_srcids, \
                                  tag_binarizer.inverse_transform(
                                      test_ts_tags_pred)):
            #ts_srcid = srcid + '_ts'
            #test_phrase_dict[ts_srcid] = test_phrase_dict[srcid] + list(ts_tags)
            #test_srcids .append(ts_srcid) # TODO: Validate if this works.
            test_phrase_dict[srcid] += list(ts_tags)
    learning_srcids += ts_learning_srcids
    manual_filter_flag = False
    if manual_filter_flag:
        for srcid in learning_srcids:
            learning_phrase_dict[srcid] = filt(learning_phrase_dict[srcid])
            learning_truths_dict[srcid] = filt(learning_truths_dict[srcid])


    ## Transform learning samples
    learning_doc = [' '.join(learning_phrase_dict[srcid]) \
                    for srcid in learning_srcids]

    test_doc = [' '.join(test_phrase_dict[srcid]) \
                for srcid in test_srcids]

    ## Augment with negative examples.
    negative_doc = []
    negative_srcids = []
    negative_truths_dict = {}
    if negative_flag:
        for srcid in learning_srcids:
            true_tagsets = list(set(learning_truths_dict[srcid]))
            sentence = learning_phrase_dict[srcid]
            for tagset in true_tagsets:
                negative_srcid = srcid + ';' + gen_uuid()
                removing_tagsets = set()
                new_removing_tagsets = set([tagset])
                removing_tags = []
                negative_tagsets = list(filter(tagset.__ne__, true_tagsets))
                i = 0
                while len(new_removing_tagsets) != len(removing_tagsets):
                    i += 1
                    if i>5:
                        pdb.set_trace()
                    removing_tagsets = deepcopy(new_removing_tagsets)
                    for removing_tagset in removing_tagsets:
                        removing_tags += removing_tagset.split('_')
                    for negative_tagset in negative_tagsets:
                        for tag in removing_tags:
                            if tag in negative_tagset.split('_'):
                                new_removing_tagsets.add(negative_tagset)
                negative_sentence = [tag for tag in sentence if\
                                     tag not in removing_tags]
                for tagset in removing_tagsets:
                    negative_tagsets = list(filter(tagset.__ne__,
                                                   negative_tagsets))

    #            negative_sentence = [word for word in sentence \
    #                                 if word not in tagset.split('_')]
                negative_doc.append(' '.join(negative_sentence))
                negative_truths_dict[negative_srcid] = negative_tagsets
                negative_srcids.append(negative_srcid)
        for i in range(0,50):
            # Add empty examples
            negative_srcid = gen_uuid()
            negative_srcids.append(negative_srcid)
            negative_doc.append('')
            negative_truths_dict[negative_srcid] = []

    #learning_doc += negative_doc
    #learning_srcids += negative_srcids
    #learning_truths_dict.update(negative_truths_dict)


    ## Init Brick document
    brick_truths_dict = dict()
    brick_srcids = []
    brick_doc = []
    if use_brick_flag:
        logging.info('Start adding Brick samples')
        #brick_copy_num = int(len(learning_phrase_dict) * 0.04)
        #if brick_copy_num < 4:
        #brick_copy_num = 4
        #brick_copy_num = 2
        brick_copy_num = 6
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
                #multiplier = random.randint(2, 6)
                srcid = 'brick;' + gen_uuid()
                brick_srcids.append(srcid)
                brick_truths_dict[srcid] = [tagset]
                tagset_doc = list()
                for tag in tagset.split('_'):
                    tagset_doc += [tag] * random.randint(1,2)
                brick_doc.append(' '.join(tagset_doc))

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
        logging.info('Finished adding Brick samples')
    #brick_truth_mat = csr_matrix([tagset_binarizer.transform(\
    #                              [brick_truths_dict[srcid]])[0] \
    #                              for srcid in brick_srcids])

    logging.info('start tagset vectorizing')
    tagset_vectorizer.fit(learning_doc + test_doc)# + brick_doc)
    logging.info('finished tagset vectorizing')

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
    #learning_doc += brick_doc
    #learning_srcids += brick_srcids
    #learning_truths_dict.update(brick_truths_dict)

    """
    raw_learning_vect_doc = tagset_vectorizer.transform(learning_doc)
    learning_vect_doc = raw_learning_vect_doc.toarray()
    if eda_flag:
        for building in source_target_buildings:
            building_mask = np.array([1 if srcid in brick_srcids \
                                or find_key(srcid, total_srcid_dict, check_in)\
                                    == building\
                                        else 0 for srcid in learning_srcids])
            pdb.set_trace()
            learning_vect_doc = np.hstack([learning_vect_doc] \
                                 + [np.asmatrix(building_mask \
                                    * learning_vect.toarray()[0]).T \
                                for learning_vect in raw_learning_vect_doc.T])
    """
    if eda_flag:
        learning_vect_doc = tagset_vectorizer.transform(learning_doc +
                                                        negative_doc).todense()
        learning_srcids += negative_srcids
        new_learning_vect_doc = deepcopy(learning_vect_doc)
        for building in source_target_buildings:
            building_mask = np.array([1 if find_key(srcid.split(';')[0],\
                                                   total_srcid_dict,\
                                                   check_in) == building
                                      else 0 for srcid in learning_srcids])
            new_learning_vect_doc = np.hstack([new_learning_vect_doc] \
                                 + [np.asmatrix(building_mask \
                                    * np.asarray(learning_vect)[0]).T \
                                for learning_vect \
                                    in learning_vect_doc.T])
        learning_vect_doc = new_learning_vect_doc
        if use_brick_flag:
            new_brick_srcids = list()
            new_brick_vect_doc = np.array([])\
                    .reshape((0, len(tagset_vectorizer.vocabulary) \
                              * (len(source_target_buildings)+1)))
            brick_vect_doc = tagset_vectorizer.transform(brick_doc).todense()
            for building in source_target_buildings:
                prefixer = lambda srcid: building + '-' + srcid
                one_brick_srcids = list(map(prefixer, brick_srcids))
                for new_brick_srcid, brick_srcid\
                        in zip(one_brick_srcids, brick_srcids):
                    brick_truths_dict[new_brick_srcid] = \
                            brick_truths_dict[brick_srcid]
                one_brick_vect_doc = deepcopy(brick_vect_doc)
                for b in source_target_buildings:
                    if b != building:
                        one_brick_vect_doc = np.hstack([
                            one_brick_vect_doc,
                            np.zeros((len(brick_srcids),
                                      len(tagset_vectorizer.vocabulary)))])
                    else:
                        one_brick_vect_doc = np.hstack([
                            one_brick_vect_doc, brick_vect_doc])
                new_brick_vect_doc = np.vstack([new_brick_vect_doc,
                                            one_brick_vect_doc])
                new_brick_srcids += one_brick_srcids
            learning_vect_doc = np.vstack([learning_vect_doc,
                                           new_brick_vect_doc])
            brick_srcids = new_brick_srcids
            learning_srcids += brick_srcids

    else:
        learning_vect_doc = tagset_vectorizer.transform(learning_doc +
                                                        negative_doc +
                                                        brick_doc).todense()
        learning_srcids += negative_srcids + brick_srcids
    learning_truths_dict.update(negative_truths_dict)
    learning_truths_dict.update(brick_truths_dict)

    truth_mat = csr_matrix([tagset_binarizer.transform(\
                    [learning_truths_dict[srcid]])[0]\
                        for srcid in learning_srcids])
    point_truths_dict = dict()
    point_srcids = list()
    for srcid in learning_srcids:
    #for srcid, truths in learning_truths_dict.items():
        truths = learning_truths_dict[srcid]
        point_tagset = None
        for tagset in truths:
            if tagset in point_tagsets:
                point_tagset = tagset
                break
        if point_tagset:
            point_truths_dict[srcid] = point_tagset
            point_srcids.append(srcid)

    try:
        point_truth_mat = [point_tagsets.index(point_truths_dict[srcid]) \
                           for srcid in point_srcids]
        point_vect_doc = np.vstack([learning_vect_doc[learning_srcids.index(srcid)]
                                    for srcid in point_srcids])
    except:
        pdb.set_trace()
    if eda_flag:
        zero_vectors = tagset_binarizer.transform(\
                    [[] for i in range(0, unlabeled_vect_doc.shape[0])])
        truth_mat = vstack([truth_mat, zero_vectors])
        learning_vect_doc = np.vstack([learning_vect_doc, unlabeled_vect_doc])
    logging.info('Start learning multi-label classifier')

    ## FITTING A CLASSIFIER
    if tagset_classifier_type == 'RandomForest':
        def meta_rf(**kwargs):
            #return RandomForestClassifier(**kwargs)
            return RandomForestClassifier(n_jobs=n_jobs, n_estimators=150)

        #tagset_classifier = RandomForestClassifier(n_estimators=100,
        #                                           random_state=0,\
        #                                           n_jobs=n_jobs)
        meta_classifier = meta_rf
        params_list_dict = {}
    elif tagset_classifier_type == 'StructuredCC_BACKUP':
        #feature_selector = SelectFromModel(LinearSVC(C=0.001))
        feature_selector = SelectFromModel(LinearSVC(C=0.01, penalty='l1', dual=False))
        base_base_classifier = PassiveAggressiveClassifier(loss='squared_hinge', C=0.1)
        #base_base_classifier = GradientBoostingClassifier()
        #base_base_classifier = RandomForestClassifier()
        base_classifier = Pipeline([('feature_selection',
                                     feature_selector),
                                    ('classification',
                                     base_base_classifier)
                                   ])
        tagset_classifier = StructuredClassifierChain(
                                base_classifier,
                                tagset_binarizer,
                                subclass_dict,
                                tagset_vectorizer.vocabulary,
                                n_jobs,
                                use_brick_flag)
    elif tagset_classifier_type == 'Project':
        def meta_proj(**kwargs):
            #base_classifier = LinearSVC(C=20, penalty='l1', dual=False)
            base_classifier = SVC(kernel='rbf', C=10, class_weight='balanced')
            #base_classifier = GaussianProcessClassifier()
            tagset_classifier = ProjectClassifier(base_classifier,
                                                           tagset_binarizer,
                                                           tagset_vectorizer,
                                                           subclass_dict,
                                                           n_jobs)
            return tagset_classifier
        meta_classifier = meta_proj
        params_list_dict = {}


    elif tagset_classifier_type == 'StructuredCC':
        def meta_scc(**kwargs):
            feature_selector = SelectFromModel(LinearSVC(C=1))
            #feature_selector = SelectFromModel(LinearSVC(C=0.01, penalty='l1', dual=False))
            base_base_classifier = GradientBoostingClassifier(**kwargs)
            #base_base_classifier = SGDClassifier(loss='modified_huber', penalty='elasticnet')
            #base_base_classifier = PassiveAggressiveClassifier(loss='squared_hinge', C=0.1)
            #base_base_classifier = LogisticRegression()
            #base_base_classifier = RandomForestClassifier(**kwargs)
            base_classifier = Pipeline([('feature_selection',
                                         feature_selector),
                                        ('classification',
                                         base_base_classifier)
                                       ])
            tagset_classifier = StructuredClassifierChain(
                                base_classifier,
                                tagset_binarizer,
                                subclass_dict,
                                tagset_vectorizer.vocabulary,
                                n_jobs,
                                use_brick_flag,
                                tagset_vectorizer)
            return tagset_classifier
        meta_classifier = meta_scc
        rf_params_list_dict = {
            'n_estimators': [10, 50, 100],
            'criterion': ['gini', 'entropy'],
            'max_features': [None, 'auto'],
            'max_depth': [1, 5, 10, 50],
            'min_samples_leaf': [2,4,8],
            'min_samples_split': [2,4,8]
        }
        gb_params_list_dict = {
            'loss': ['deviance', 'exponential'],
            'learning_rate': [0.1, 0.01, 1, 2],
            'criterion': ['friedman_mse', 'mse'],
            'max_features': [None, 'sqrt'],
            'max_depth': [1, 3, 5, 10],
            'min_samples_leaf': [1,2,4,8],
            'min_samples_split': [2,4,8]
        }
        params_list_dict = gb_params_list_dict
    elif tagset_classifier_type == 'StructuredCC_LinearSVC':
        base_classifier = LinearSVC(loss='hinge', tol=1e-5,\
                                    max_iter=2000, C=2,
                                    fit_intercept=False,
                                    class_weight='balanced')
        tagset_classifier = StructuredClassifierChain(base_classifier,
                                                      tagset_binarizer,
                                                      subclass_dict,
                                                      tagset_vectorizer.vocabulary,
                                                      n_jobs)
    elif tagset_classifier_type == 'OneVsRest':
        base_classifier = LinearSVC(loss='hinge', tol=1e-5,\
                                    max_iter=2000, C=2,
                                    fit_intercept=False,
                                    class_weight='balanced')
        tagset_classifier = OneVsRestClassifier(base_classifier)
    elif tagset_classifier_type == 'Voting':
        def meta_voting(**kwargs):
            return VotingClassifier(tagset_binarizer, tagset_vectorizer,
                                    tagset_tree, tagset_list)
        meta_classifier = meta_voting
        params_list_dict = {}
    else:
        assert False

    if not isinstance(truth_mat, csr_matrix):
        truth_mat = csr_matrix(truth_mat)

    tagset_classifier = parameter_validation(learning_vect_doc[:orig_sample_num],
                         truth_mat[:orig_sample_num],
                         orig_learning_srcids,
                         params_list_dict, meta_classifier, tagset_vectorizer,
                         tagset_binarizer, source_target_buildings, eda_flag)

    if isinstance(tagset_classifier, StructuredClassifierChain):
        tagset_classifier.fit(learning_vect_doc, truth_mat.toarray(), \
                              orig_sample_num=len(learning_vect_doc)
                              - len(brick_srcids))
    else:
        tagset_classifier.fit(learning_vect_doc, truth_mat.toarray())
    point_classifier.fit(point_vect_doc, point_truth_mat)
    logging.info('Finished learning multi-label classifier')

    return tagset_classifier, tagset_vectorizer, tagset_binarizer, \
            point_classifier, ts2ir


def cross_validation(building_list, n_list,
                     target_building, \
                     learning_srcids, test_srcids, \
                     pred_phrase_dict, pred_tagsets_dict,
                     result_dict, k=2, \
                     eda_flag=False, token_type='justseparate',
                     use_brick_flag=True,
                     ts_flag=False):

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
        logging.info('Validation iteration: {0} / {1}'\
                     .format(cid, len(cluster_dict)))
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


        source_target_buildings = list(set(building_list + [target_building]))
        tagset_classifier, tagset_vectorizer, tagset_binarizer, \
                point_classifier, ts2ir = \
            build_tagset_classifier(building_list_1, validation_building,\
                            validation_sentence_dict, \
                            validation_token_label_dict,\
                            phrase_dict_1, phrase_dict_2,\
                            truths_dict_1,\
                            chosen_learning_srcids+cluster_srcids, \
                            validation_srcids,\
                            tagset_list, eda_flag, use_brick_flag,
                            source_target_buildings,
                            4,
                            ts_flag
                           )

        validation_pred_tagsets_dict, \
        validation_pred_certainty_dict, \
        pred_point_dict = \
                tagsets_prediction(tagset_classifier, tagset_vectorizer, \
                                   tagset_binarizer, validation_phrase_dict, \
                                   validation_srcids, source_target_buildings,
                                  eda_flag, point_classifier)

        curr_result_dict = tagsets_evaluation(validation_truths_dict, \
                                         validation_pred_tagsets_dict, \
                                         validation_pred_certainty_dict,\
                                         validation_srcids,
                                         pred_point_dict,
                                         validation_phrase_dict,
                                             )

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
            'common_pred_points_len': len(pred_found_points),
            'common_true_points_len': len(true_found_points),
            'point_precision': precision_dict[cid],
            'point_recall': recall_dict[cid]
        }
        mem_usage = psutil.virtual_memory().used/1000/1000/1000
        logging.info('Current memory usage {0} GB'.format(mem_usage))
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
               and cluster_result['common_pred_points_len'] > 1:
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
                                         ts_flag=False,
                                         negative_flag=True,
                                         n_jobs=4,
                                         prev_step_data={
                                             'learning_srcids':[],
                                             'iter_cnt':0,
                                             'point_precision_history': [],
                                             'point_recall_history':  [],
                                             'correct_point_cnt_history': [],
                                             'incorrect_point_cnt_history': [],
                                             'unfound_point_cnt_history': [],
                                             'subset_accuracy_history':  [],
                                             'accuracy_history': [],
                                             'hamming_loss_history': [],
                                             'hierarchy_accuracy_history': [],
                                             'weighted_f1_history': [],
                                             'macro_f1_history': [],
                                             'metadata': {},
                                             'phrase_usage_history': []
                                         },
                                         crf_phrase_dict=None,
                                         crf_srcids=None
                                        ):
    logging.info('Entity Recognition Get Started.')
    global tagset_list
    global total_srcid_dict
    global tree_depth_dict
    inc_num = 20
    assert len(building_list) == len(source_sample_num_list)

    ########################## DATA INITIATION ##########################
    # construct source data information data structure
    source_cnt_list = [[building, cnt]\
                       for building, cnt\
                       in zip(building_list, source_sample_num_list)]
    if not prev_step_data['metadata']:
        metadata = {
            'token_type': token_type,
            'label_type': label_type,
            'use_cluster_flag': use_cluster_flag,
            'use_brick_flag': use_brick_flag,
            'eda_flag': eda_flag,
            'ts_flag': ts_flag,
            'negative_flag': negative_flag,
            'building_list': building_list,
            'target_building': target_building,
            'source_sample_num_list': source_sample_num_list,
        }
        prev_step_data['metadata'] = metadata
    if not prev_step_data.get('learnt_but_unfound_tagsets_history'):
        prev_step_data['learnt_but_unfound_tagsets_history'] = list()
    if not prev_step_data.get('unfound_tagsets_history'):
        prev_step_data['unfound_tagsets_history'] = []

    prev_step_data['metadata']['inc_num'] =  inc_num
    if not prev_step_data.get('debug'):
        prev_step_data['debug'] = defaultdict(list)

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
        """
        with open('metadata/{0}_char_label_dict.json'\
                  .format(target_building), 'r') as fp:
            sentence_label_dict = json.load(fp)
        test_srcids = [srcid for srcid in sentence_label_dict.keys() \
                       if srcid not in learning_srcids]
        todo_srcids = select_random_samples(target_building,
                                            test_srcids,
                                            int(inc_num/2),
                                            use_cluster_flag,
                                            token_type='justseparate',
                                            reverse=True)
        todo_srcids += select_random_samples(target_building,
                                            test_srcids,
                                            int(inc_num/2),
                                            use_cluster_flag,
                                            token_type='justseparate',
                                            reverse=False)
        #learning_srcids = cross_validation(building_list, source_sample_num_list, \
        #                 target_building, \
        #                 learning_srcids, prev_test_srcids, \
        #                 prev_pred_phrase_dict, prev_pred_tagsets_dict,
        #                 prev_result_dict, 4, \
        #                 eda_flag, token_type, use_brick_flag, ts_flag)
        learning_srcids += todo_srcids * 5
        """
    print('\n')
    print('################ Iteration {0} ################'.format(iter_cnt))

    ### Get Learning Data
    sample_srcid_list_dict = dict()
    validation_srcids = []
    validation_truths_dict = {}
    for building, sample_num in zip(building_list, source_sample_num_list):
        with open('metadata/{0}_char_label_dict.json'\
                  .format(building), 'r') as fp:
            sentence_label_dict = json.load(fp)
        srcids = list(sentence_label_dict.keys())
        if iter_cnt == 1:
            sample_srcid_list = select_random_samples(\
                                    building,\
                                    sentence_label_dict.keys(),\
                                    sample_num, \
                                    use_cluster_flag,\
                                    token_type=token_type,
                                    reverse=True,
                                    shuffle_flag=False)
            sample_srcid_list_dict[building] = sample_srcid_list
            learning_srcids += sample_srcid_list
            total_srcid_dict[building] = list(sentence_label_dict.keys())
        else:
            sample_srcid_list_dict[building] = [srcid for srcid\
                                                in srcids \
                                                if srcid in learning_srcids]
        validation_num = min(len(sentence_label_dict)
                             - len(sample_srcid_list_dict[building]),
                             len(sample_srcid_list_dict[building]))
        validation_srcids += select_random_samples(\
                                    building,\
                                    srcids,\
                                    validation_num,\
                                    use_cluster_flag,\
                                    token_type=token_type,
                                    reverse=True,
                                    shuffle_flag=False)
    
    _, \
    _, \
    validation_truths_dict, \
    _ = get_multi_buildings_data(building_list, \
                                           validation_srcids, \
                                           eda_flag,\
                                           token_type)

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
    tree_depth_dict = calc_leaves_depth(tagset_tree)
    assert tree_depth_dict['supply_air_static_pressure_integral_time_setpoint']\
            > 3

    source_target_buildings = list(set(building_list + [target_building]))
    begin_time = arrow.get()
    tagset_classifier, tagset_vectorizer, tagset_binarizer, \
            point_classifier, ts2ir = \
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
                            n_jobs,
                            ts_flag,
                            negative_flag,
                            validation_truths_dict
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
                                   tagset_binarizer, \
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
                                tagset_binarizer, \
                                phrase_dict, \
                                sorted(learning_srcids),\
                                building_list,\
                                eda_flag,\
                                point_classifier,
                                ts2ir)
    eval_learning_srcids = deepcopy(learning_srcids)
    random.shuffle(eval_learning_srcids)
    learning_result_dict = tagsets_evaluation(learning_truths_dict,
                                         learning_pred_tagsets_dict,
                                         learning_pred_certainty_dict,
                                         eval_learning_srcids,
                                         learning_pred_point_dict,\
                                         phrase_dict,\
                                         debug_flag,
                                         tagset_classifier,
                                         tagset_vectorizer)

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
                                tagset_binarizer, \
                                target_phrase_dict, \
                                target_srcids,\
                                source_target_buildings,\
                                eda_flag,\
                                point_classifier,
                                ts2ir)
    eval_target_srcids = deepcopy(target_srcids)
    random.shuffle(eval_target_srcids)
    target_result_dict = tagsets_evaluation(target_truths_dict, \
                                         target_pred_tagsets_dict, \
                                         target_pred_certainty_dict,\
                                         eval_target_srcids,\
                                         target_pred_point_dict,\
                                         target_phrase_dict,\
                                            debug_flag,
                                           tagset_classifier,
                                           tagset_vectorizer)

    learnt_tagsets = Counter(reduce(adder, [learning_truths_dict[srcid]
                                    for srcid in learning_srcids]))
    unfound_tagsets = list()
    for srcid in target_srcids:
        pred_tagsets = target_pred_tagsets_dict[srcid]
        true_tagsets = target_truths_dict[srcid]
        for tagset in true_tagsets:
            if tagset not in pred_tagsets:
                unfound_tagsets.append(tagset)
    unfound_tagsets = Counter(unfound_tagsets)
    learnt_unfound_tagsets = dict([(tagset, learnt_tagsets[tagset])
                 for tagset in unfound_tagsets.keys()
                 if tagset in learnt_tagsets.keys()])

    next_step_data = {
        'debug': prev_step_data['debug'],
        'exp_type': 'entity_from_ground_truth',
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
            + [target_result_dict['unfound_point_cnt']],
        'subset_accuracy_history': \
            prev_step_data['subset_accuracy_history'] \
            + [target_result_dict['subset_accuracy']],
        'accuracy_history': \
            prev_step_data['accuracy_history'] \
            + [target_result_dict['accuracy']],
        'hierarchy_accuracy_history': \
            prev_step_data['hierarchy_accuracy_history'] \
            + [target_result_dict['hierarchy_accuracy']],
        'hamming_loss_history': \
            prev_step_data['hamming_loss_history'] \
            + [target_result_dict['hamming_loss']],
        'weighted_f1_history': \
            prev_step_data['weighted_f1_history'] \
            + [target_result_dict['weighted_f1']],
        'macro_f1_history': \
            prev_step_data['macro_f1_history'] \
            + [target_result_dict['macro_f1']],
        'metadata': prev_step_data['metadata'],
        'phrase_usage_history': prev_step_data['phrase_usage_history']
                                 + [target_result_dict['phrase_usage']],
        'learnt_but_unfound_tagsets_history':
            prev_step_data['learnt_but_unfound_tagsets_history'] +
            [list(learnt_unfound_tagsets.keys())],
        'unfound_tagsets_history':
            prev_step_data['unfound_tagsets_history'] +
            [list(unfound_tagsets.keys())]
    }
    print('################################# Iter# {0}'.format(iter_cnt))
    #logging.info('Learnt but not detected tagsets: {0}'.format(
    #    learnt_unfound_tagsets))
    lengther = lambda x: len(x)
    logging.info('# of not detected tagsets: {0}'.format(list(map(lengther,
                    next_step_data['unfound_tagsets_history']))))
    logging.info('# of learnt but not detected tagsets: {0}'.format(list(map(lengther,
                    next_step_data['learnt_but_unfound_tagsets_history']))))

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
    print('history of accuracy: {0}'\
          .format(next_step_data['accuracy_history']))
    print('history of micro f1: {0}'\
          .format(next_step_data['macro_f1_history']))

    # Post processing to select next step learning srcids
    phrase_usages = list(target_result_dict['phrase_usage'].values())
    mean_usage_rate = np.mean(phrase_usages)
    std_usage_rate = np.std(phrase_usages)
    threshold = mean_usage_rate - std_usage_rate
    todo_sentence_dict = dict((srcid, alpha_tokenizer(''.join(\
                                                test_sentence_dict[srcid]))) \
                              for srcid, usage_rate \
                              in target_result_dict['phrase_usage'].items() \
                              if usage_rate<threshold and srcid in test_srcids)
    try:
        cluster_srcid_dict = hier_clustering(todo_sentence_dict, threshold=2)
    except:
        cluster_srcid_dict = hier_clustering(test_sentence_dict, threshold=2)
    todo_srcids = select_random_samples(target_building, \
                          list(todo_sentence_dict.keys()),
                          min(inc_num, len(todo_sentence_dict)), \
                          use_cluster_flag,\
                          token_type=token_type,
                          reverse=True,
                          cluster_dict=cluster_srcid_dict,
                          shuffle_flag=False
                         )
    tot_todo_tagsets =  Counter(reduce(adder, [test_truths_dict[srcid] for srcid in todo_srcids], []))
    correct_todo_srcids = list()
    for srcid in todo_srcids:
        todo_tagsets = test_truths_dict[srcid]
        for tagset in todo_tagsets:
            if tagset in unfound_tagsets.keys():
                correct_todo_srcids.append(srcid)
                break
    unfound_todo_tagsets = [tagset for tagset in unfound_tagsets.keys()\
                            if tagset not in tot_todo_tagsets.keys()]
    found_todo_tagsets = [tagset for tagset in tot_todo_tagsets.keys()\
                          if tagset in unfound_tagsets]

    logging.info('# total found tagsets in todo: {0}'.format(len(tot_todo_tagsets)))
    logging.info('# unfound tagsets: {0}'.format(len(unfound_tagsets)))
    logging.info('# todo tagsets accounting unfound tagsets: {0}'.format(len(found_todo_tagsets)))
    try:
        if len(todo_srcids) > 0:
            todo_srcid_rate = len(correct_todo_srcids) / len(todo_srcids)
        else:
            todo_srcid_rate = 0
    except:
        pdb.set_trace()
    try:
        if len(tot_todo_tagsets) > 0:
            todo_tagset_rate = len(found_todo_tagsets) / len(tot_todo_tagsets)
        else:
            todo_tagset_rate = 0
    except:
        pdb.set_trace()
    logging.info('Ratio of correctly found todo srcids: {0}'.format(todo_srcid_rate))
    next_step_data['debug']['#tagsets_in_todo'].append(len(tot_todo_tagsets))
    next_step_data['debug']['#unfound_tagsets'].append(len(unfound_tagsets))
    next_step_data['debug']['#correct_tagsts_in_todo'].append(len(found_todo_tagsets))
    next_step_data['debug']['rate_correct_todo_srcids'].append(todo_srcid_rate)
    next_step_data['debug']['rate_correct_todo_tagsets'].append(todo_tagset_rate)
    next_step_data['debug']['rate_correct_todo_tagsets'].append(todo_tagset_rate)

    next_step_data['learning_srcids'] = learning_srcids + todo_srcids * 3

    try:
        del next_step_data['_id']
    except:
        pass
    with open('result/entity_iteration.json', 'w') as fp:
        json.dump(next_step_data, fp, indent=2)
    return target_result_dict, next_step_data


def find_todo_srcids(usage_rate_dict, srcids, inc_num, sentence_dict,
                     target_building):
    # Post processing to select next step learning srcids
    usages = list(usage_rate_dict.values())
    mean_usage_rate = np.mean(usages)
    std_usage_rate = np.std(usages)
    threshold = mean_usage_rate - std_usage_rate
    todo_sentence_dict = dict((srcid, alpha_tokenizer(''.join(\
                                                sentence_dict[srcid]))) \
                              for srcid, usage_rate \
                              in usage_rate_dict.items()
                              if usage_rate<threshold and srcid in srcids)
    try:
        cluster_srcid_dict = hier_clustering(todo_sentence_dict, threshold=2)
    except:
        cluster_srcid_dict = hier_clustering(test_sentence_dict, threshold=2)
    todo_srcids = select_random_samples(target_building, \
                          list(todo_sentence_dict.keys()),
                          min(inc_num, len(todo_sentence_dict)), \
                          True,\
                          token_type='justseparate',
                          reverse=True,
                          cluster_dict=cluster_srcid_dict,
                          shuffle_flag=False
                         )
    return todo_srcids


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
        'subset_accuracy_history':  [],
        'accuracy_history': [],
        'hamming_loss_history': [],
        'hierarchy_accuracy_history': [],
        'weighted_f1_history': [],
        'macro_f1_history': [],
        'metadata': {},
        'phrase_usage_history': []
    }
    for i in range(0, iter_num):
        _, step_data = entity_recognition_from_ground_truth(\
                              building_list = args[0],\
                              source_sample_num_list = args[1],\
                              target_building = args[2],\
                              token_type = args[3],\
                              label_type = args[4],\
                              use_cluster_flag = args[5],\
                              use_brick_flag = args[6],\
                              debug_flag = args[7],\
                              eda_flag = args[8],\
                              ts_flag = args[9], \
                              negative_flag = args[10],
                              n_jobs = args[11],\
                              prev_step_data = step_data
                            )
    store_result(step_data)

def iteration_wrapper(iter_num, func, *args):
    step_data = {
        'learning_srcids': [],
        'iter_cnt': 0,
    }
    step_datas = list()
    prev_data = {'iter_num':0}
    for i in range(0, iter_num):
        step_data = func(prev_data, *args)
        step_datas.append(step_data)
        prev_data = step_data
        prev_data['iter_num'] += 1
    return step_datas

def crf_entity_recognition_iteration(iter_num, postfix, *args):
    building_list = args[0]
    target_building = args[2]
    step_datas = iteration_wrapper(iter_num, entity_recognition_from_crf, *args)
    with open('result/crf_entity_iter_{0}_{1}.json'\
            .format(''.join(building_list+[target_building]), postfix), 'w') as fp:
#with open('result/crf_entity_iter_{0}_2.json'.format(''.join(building_list+[target_building])), 'w') as fp:
        json.dump(step_datas, fp, indent=2)


def determine_used_phrases(phrases, tagsets):
    phrases_usages = list()
    pred_tags = reduce(adder, [tagset.split('_') for tagset in tagsets], [])
    used_cnt = 0.0
    unused_cnt = 0.0
    for phrase in phrases:
        phrase_tags = phrase.split('_')
        for tag in phrase_tags:
            if tag in ['leftidentifier', 'rightidentifier']:
                continue
            if tag in pred_tags:
                used_cnt += 1 / len(phrase_tags)
            else:
                unused_cnt += 1 / len(phrase_tags)
    if used_cnt == 0:
        score = 0
    else:
        score = used_cnt / (used_cnt + unused_cnt) 
    return score


def determine_used_tokens(sentence, token_labels, tagsets):
    token_usages = list()
    tags = reduce(adder, [tagset.split('_') for tagset in tagsets], [])
    for token, label in zip(sentence, token_labels):
        if label=='O':
            token_usages.append(0)
        else:
            tags_in_label = label[2:].split('_')
            if tags_in_label[0] in ['leftidentifier',
                                    'rightidentifier']:
                continue
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



def entity_recognition_from_crf(prev_step_data,\
                                building_list,\
                                source_sample_num_list,\
                                target_building,\
                                token_type='justseparate',\
                                label_type='label',\
                                use_cluster_flag=False,\
                                use_brick_flag=True,\
                                eda_flag=False,\
                                negative_flag = True,\
                                debug_flag=False,
                                n_jobs=4,
                                ts_flag=False):

    global tagset_list
    global total_srcid_dict
    global tagset_tree
    global tree_depth_dict
    inc_num = 20

    ### Initialize CRF Data
    crf_result_query = {
        'label_type': label_type,
        'token_type': token_type,
        'use_cluster_flag': use_cluster_flag,
        'building_list': building_list,
        'source_sample_num_list': source_sample_num_list,
        'target_building': target_building,
    }
    if prev_step_data.get('learning_srcids_history'):
        crf_result_query['learning_srcids'] = prev_step_data['learning_srcids_history'][-1]


    # TODO: Make below to learn if not exists
    crf_result = get_crf_results(crf_result_query)
    if not crf_result:
        if crf_result_query.get('learning_srcids'):
            learning_srcids = sorted(crf_result_query['learning_srcids'])
        else:
            learning_srcids = []

        learn_crf_model(building_list,
                    source_sample_num_list,
                    token_type,
                    label_type,
                    use_cluster_flag,
                    debug_flag,
                    use_brick_flag,
                    {
                        'learning_srcids': deepcopy(learning_srcids),
                        'iter_cnt':0
                    })
        crf_test(building_list,
                 source_sample_num_list,
                 target_building,
                 token_type,
                 label_type,
                 use_cluster_flag,
                 use_brick_flag,
                 learning_srcids)
        crf_result = get_crf_results(crf_result_query)
        assert crf_result
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
    source_target_buildings = list(set(building_list + [target_building]))
    classifier, vectorizer, binarizer, point_classifier, ts2ir = \
            build_tagset_classifier(building_list, target_building,\
                                    crf_sentence_dict, crf_token_label_dict,\
                                    given_phrase_dict, crf_phrase_dict,\
                                    given_truths_dict,\
                                    given_srcids, crf_srcids,\
                                    tagset_list, eda_flag, use_brick_flag,
                                    source_target_buildings,
                                    n_jobs,
                                    ts_flag,
                                    negative_flag
                                   )
    augment_tagset_tree(tagset_list)
    tree_depth_dict = calc_leaves_depth(tagset_tree)

    crf_pred_tagsets_dict, \
    crf_pred_certainty_dict, \
    crf_pred_point_dict = tagsets_prediction(\
                                   classifier, vectorizer, \
                                   binarizer, crf_phrase_dict, \
                                   crf_srcids, source_target_buildings,
                                   eda_flag, point_classifier, ts2ir)

    crf_token_usage_dict = determine_used_tokens_multiple(\
                                crf_sentence_dict, crf_token_label_dict, \
                                crf_pred_tagsets_dict, crf_srcids)

    crf_token_usage_rate_dict = dict((srcid, sum(usage)/len(usage))\
                                     for srcid, usage \
                                     in crf_token_usage_dict.items())

    crf_entity_result_dict = tagsets_evaluation(crf_truths_dict, crf_pred_tagsets_dict,
                       crf_pred_certainty_dict, crf_srcids,
                       crf_pred_point_dict, crf_phrase_dict, debug_flag)

    usage_rates = list(crf_token_usage_rate_dict.values())
    usage_rate_mean = np.mean(usage_rates)
    usage_rate_mean = np.std(usage_rates)

    #todo_sentence_dict = dict((srcid, alpha_tokenizer(''.join(\
    #                                            crf_sentence_dict[srcid]))) \
    #                          for srcid, usage_rate \
    #                          in crf_token_usage_rate_dict.items() \
    #                          if usage_rate<0.3)
    """
    cluster_srcid_dict = hier_clustering(todo_sentence_dict, threshold=2)
    """
    #TODO: Implement this!!!!!!!!
    todo_srcids = find_todo_srcids(crf_token_usage_rate_dict, crf_srcids, inc_num, crf_sentence_dict,
                     target_building)
    #unknown_srcids = todo_sentence_dict.keys()
    #todo_srcids = select_random_samples(target_building, unknown_srcids, \
    #                                    len(unknown_srcids)*0.1, True)
    curr_learning_srcids = sorted(reduce(adder, crf_result['source_list']\
                             .values()))
    updated_learning_srcids = todo_srcids \
                            + curr_learning_srcids
    del crf_result['result']
    del crf_result['_id']
    next_step_data = prev_step_data
    if not next_step_data.get('result'):
        next_step_data['result'] = dict()
    if not next_step_data['result'].get('crf'):
        next_step_data['result']['crf'] = []
    if not next_step_data['result'].get('entity'):
        next_step_data['result']['entity'] = []
    if not next_step_data.get('learning_srcids_history'):
        next_step_data['learning_srcids_history'] = [curr_learning_srcids]

    next_step_data['iter_num'] += 1
    next_step_data['learning_srcids_history'].append(updated_learning_srcids)
    next_step_data['result']['entity'].append(crf_entity_result_dict)
    next_step_data['result']['crf'].append(crf_result)
    return next_step_data

def calc_char_utilization(sentence, char_labels, tagsets):
    assert len(sentence) == len(char_labels)
    tags = reduce(adder, map(splitter, tagsets))
    char_label_splitter = lambda s: splitter(s[2:])
    char_tags = reduce(adder, map(char_label_splitter, char_labels))
    used_cnt = 0
    unused_cnt = 0
    dummy_cnt = 0
    done_cnt = 0
    undone_cnt = 0
    for char_label in char_labels:
        char_tags = char_label[2:].split('_')
        for char_tag in char_tags:
            if char_tag in ['leftidentifier', 'rightidentifier', \
                            'none', 'unknown']:
                dummy_cnt += 1 / len(char_tags)
                continue
            if char_tag in tags:
                used_cnt += 1/len(char_tags)
            else:
                unused_cnt += 1/len(char_tags)
    for tagset in tagsets:
        tags = tagset.split('_')
        for tag in tags:
            if tag in char_tags:
                done_cnt += 1 / len(tags)
            else:
                undone_cnt += 1 / len(tags)
    assert done_cnt + undone_cnt == len(tagsets)
    assert used_cnt + unused_cnt + dummy_cnt == len(sentence)
    utilization = used_cnt / (used_cnt + unused_cnt)
    completeness = done_cnt / (done_cnt + undone_cnt)
    return utilization, completeness



#TODO: Make this more generic to apply to other functions
def entity_recognition_from_ground_truth_get_avg(N,
                                                 building_list,
                                                 source_sample_num_list,
                                                 target_building,
                                                 token_type='justseparate',
                                                 label_type='label',
                                                 use_cluster_flag=False,
                                                 use_brick_flag=False,
                                                 eda_flag=False,
                                                 ts_flag=False,
                                                 negative_flag=True,
                                                 n_jobs=4,
                                                 worker_num=2):

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
            eda_flag,
            ts_flag,
            negative_flag,
            n_jobs
           )

    for i in range(0,N):
        p = Process(target=parallel_func, args=(\
                entity_recognition_from_ground_truth, i, return_dict, *args))
        jobs.append(p)
        p.start()
        if i % worker_num == worker_num-1:
            for proc in jobs:
                proc.join()
            jobs = []

    for proc in jobs:
        proc.join()

    ig = lambda k, d: d[0][k]
    point_precision = np.mean(list(map(partial(ig, 'point_precision'),
                                  return_dict.values())))
    point_recall = np.mean(list(map(partial(ig, 'point_recall'),
                                  return_dict.values())))
    accuracy = np.mean(list(map(partial(ig, 'accuracy'),
                                  return_dict.values())))
    subset_accuracy = np.mean(list(map(partial(ig, 'subset_accuracy'),
                                  return_dict.values())))
    hierarchy_accuracy = np.mean(list(map(partial(ig, 'hierarchy_accuracy'),
                                  return_dict.values())))
    macro_f1 = np.mean(list(map(partial(ig, 'macro_f1'),
                                  return_dict.values())))
    print(args)
    print ('Averaged Point Precision: {0}'.format(point_precision))
    print ('Averaged Point Recall: {0}'.format(point_recall))
    print ('Averaged Subset Accuracy: {0}'.format(subset_accuracy))
    print ('Averaged Accuracy: {0}'.format(accuracy))
    print ('Averaged Hierarchy Accuracy: {0}'.format(hierarchy_accuracy))
    print ('Averaged Macro F1: {0}'.format(macro_f1))
    print("FIN")

def oxer(b):
    if b:
        return 'O'
    else:
        return 'X'

def entity_ts_result():
    source_target_list = [('ebu3b', 'ap_m')]
    n_list_list = [(200,5)]
    ts_flag = False
    eda_flag = False
    inc_num = 20
    iter_num = 10
    default_query = {
        'metadata.label_type': 'label',
        'metadata.token_type': 'justseparate',
        'metadata.use_cluster_flag': True,
        'metadata.building_list' : [],
        'metadata.source_sample_num_list': [],
        'metadata.target_building': '',
        'metadata.ts_flag': ts_flag,
        'metadata.eda_flag': eda_flag,
        'metadata.use_brick_flag': True,
        'metadata.negative_flag': True,
        'metadata.inc_num': inc_num,
    }
    query_list = [deepcopy(default_query),
                  deepcopy(default_query)]
    query_list[0]['metadata.ts_flag'] = True
    fig, ax = plt.subplots(1, len(source_target_list))
    axes = [ax]
    cs = ['firebrick', 'deepskyblue']
    for i, (ax, (source, target)) in enumerate(zip(axes, source_target_list)):
        linestyles = [':', '-.', '-']
        for query in query_list:
            for ns in n_list_list:
                if query['metadata.use_brick_flag'] and ns[0]==0:
                    continue
                n_s = ns[0]
                if i==1 and ns[1]==5:
                    n_t = 5
                else:
                    n_t = ns[1]

                if n_s == 0:
                    building_list = [target]
                    source_sample_num_list = [n_t]
                elif n_t == 0:
                    building_list = [source]
                    source_sample_num_list = [n_s]
                else:
                    building_list = [source, target]
                    source_sample_num_list = [n_s, n_t]
                query['metadata.building_list'] = building_list
                query['metadata.source_sample_num_list'] = \
                        source_sample_num_list
                query['metadata.target_building'] = target
                q = {'$and': [query, {'$where': \
                                      'this.accuracy_history.length=={0}'\
                                      .format(iter_num)}]}

                result = get_entity_results(q)
                try:
                    assert result
                except:
                    print(n_t)
                    pdb.set_trace()
                    result = get_entity_results(query)
                #point_precs = result['point_precision_history'][-1]
                #point_recall = result['point_recall'][-1]
                subset_accuracy_list = [val * 100 for val in result['subset_accuracy_history']]
                accuracy_list = [val * 100 for val in result['accuracy_history']]
                hierarchy_accuracy_list = [val * 100 for val in result['hierarchy_accuracy_history']]
                weighted_f1_list = [val * 100 for val in result['weighted_f1_history']]
                macro_f1_list = [val * 100 for val in result['macro_f1_history']]
                exp_num = len(macro_f1_list)
                target_n_list = list(range(n_t, inc_num*exp_num+1, inc_num))

                xs = target_n_list
                ys = [accuracy_list, macro_f1_list]
                #xlabel = '# of Target Building Samples'
                xlabel = None
                ylabel = 'Score (%)'
                xtick = range(0,205, 50)
                xtick_labels = [str(n) for n in xtick]
                ytick = range(0,102,20)
                ytick_labels = [str(n) for n in ytick]
                ylim = (ytick[0]-1, ytick[-1]+2)
                if i==0:
                    legends = [
                        '{0}, SA: {1}'
                        .format(n_s,
                                oxer(query['metadata.use_brick_flag'])),
                        '{0}, SA: {1}'
                        .format(n_s,
                                oxer(query['metadata.use_brick_flag']))
                    ]
                else:
                    legends = None
                title = None
                plotter.plot_multiple_2dline(xs, ys, xlabel, ylabel, xtick,\
                                 xtick_labels, ytick, ytick_labels, title, ax,\
                                 fig, ylim, None, legends, xtickRotate=0, \
                                 linestyles=[linestyles.pop()]*len(ys), cs=cs)


    for ax in axes:
        ax.grid(True)
    for ax, (source, target) in zip(axes, source_target_list):
        #ax.set_title('{0} $\Rightarrow$ {1}'.format(
        #    anon_building_dict[source], anon_building_dict[target]))
        ax.text(0.45, 0.2, '{0} $\Rightarrow$ {1}'.format(
            anon_building_dict[source], anon_building_dict[target]),
            fontsize=11,
            transform=ax.transAxes)

    for i in range(1,len(source_target_list)):
        axes[i].set_yticklabels([])
        axes[i].set_ylabel('')

    ax = axes[0]
    #handles, labels = ax.get_legend_handles_labels()
    #legend_order = [0,1,2,3,4,5]
    #new_handles = [handles[i] for i in legend_order]
    #new_labels = [labels[i] for i in legend_order]
    #ax.legend(new_handles, new_labels, bbox_to_anchor=(0.15,0.96), ncol=3, frameon=False)
    plt.text(0, 1.2, 'Accuracy: \nMacro $F_1$: ', ha='center', va='center',
            transform=ax.transAxes)
    fig.text(0.5, -0.1, '# of Target Building Samples', ha='center', 
            alpha=0)

    for i, ax in enumerate(axes):
        if i != 0:
            ax.set_xlabel('')

    fig.set_size_inches(4.4,1.5)
    save_fig(fig, 'figs/entity_ts.pdf')
    subprocess.call('./send_figures')

def entity_iter_result():
    source_target_list = [('ebu3b', 'bml'), ('ghc', 'ebu3b')]
    n_list_list = [(200,5),
                   (0,5),]
#                   (1000,1)]
    ts_flag = False
    eda_flag = False
    inc_num = 20
    iter_num = 10
    default_query = {
        'metadata.label_type': 'label',
        'metadata.token_type': 'justseparate',
        'metadata.use_cluster_flag': True,
        'metadata.building_list' : [],
        'metadata.source_sample_num_list': [],
        'metadata.target_building': '',
        'metadata.ts_flag': ts_flag,
        'metadata.eda_flag': eda_flag,
        'metadata.use_brick_flag': True,
        'metadata.negative_flag': True,
        'metadata.inc_num': inc_num,
    }
    query_list = [deepcopy(default_query),
                  deepcopy(default_query)]
    query_list[1]['metadata.negative_flag'] = False
    query_list[1]['metadata.use_brick_flag'] = False
    fig, axes = plt.subplots(1, len(source_target_list))
#    axes = [ax]
    cs = ['firebrick', 'deepskyblue']
    for i, (ax, (source, target)) in enumerate(zip(axes, source_target_list)):
        linestyles = [':', '-.', '-']
        for query in query_list:
            for ns in n_list_list:
                if query['metadata.use_brick_flag'] and ns[0]==0:
                    continue
                n_s = ns[0]
                if i==1 and ns[1]==5:
                    n_t = 5
                else:
                    n_t = ns[1]

                if n_s == 0:
                    building_list = [target]
                    source_sample_num_list = [n_t]
                elif n_t == 0:
                    building_list = [source]
                    source_sample_num_list = [n_s]
                else:
                    building_list = [source, target]
                    source_sample_num_list = [n_s, n_t]
                query['metadata.building_list'] = building_list
                query['metadata.source_sample_num_list'] = \
                        source_sample_num_list
                query['metadata.target_building'] = target
                q = {'$and': [query, {'$where': \
                                      'this.accuracy_history.length=={0}'\
                                      .format(iter_num)}]}

                result = get_entity_results(q)
                try:
                    assert result
                except:
                    print(n_t)
                    pdb.set_trace()
                    result = get_entity_results(query)
                #point_precs = result['point_precision_history'][-1]
                #point_recall = result['point_recall'][-1]
                subset_accuracy_list = [val * 100 for val in result['subset_accuracy_history']]
                accuracy_list = [val * 100 for val in result['accuracy_history']]
                hierarchy_accuracy_list = [val * 100 for val in result['hierarchy_accuracy_history']]
                weighted_f1_list = [val * 100 for val in result['weighted_f1_history']]
                macro_f1_list = [val * 100 for val in result['macro_f1_history']]
                exp_num = len(macro_f1_list)
                target_n_list = list(range(n_t, inc_num*exp_num+1, inc_num))

                xs = target_n_list
                ys = [accuracy_list, macro_f1_list]
                #xlabel = '# of Target Building Samples'
                xlabel = None
                ylabel = 'Score (%)'
                xtick = range(0,205, 50)
                xtick_labels = [str(n) for n in xtick]
                ytick = range(0,102,20)
                ytick_labels = [str(n) for n in ytick]
                ylim = (ytick[0]-1, ytick[-1]+2)
                if i==0:
                    legends = [
                        '{0}, SA: {1}'
                        .format(n_s,
                                oxer(query['metadata.use_brick_flag'])),
                        '{0}, SA: {1}'
                        .format(n_s,
                                oxer(query['metadata.use_brick_flag']))
                    ]
                else:
                    legends = None
                title = None
                plotter.plot_multiple_2dline(xs, ys, xlabel, ylabel, xtick,\
                                 xtick_labels, ytick, ytick_labels, title, ax,\
                                 fig, ylim, None, legends, xtickRotate=0, \
                                 linestyles=[linestyles.pop()]*len(ys), cs=cs)
            pdb.set_trace()


    for ax in axes:
        ax.grid(True)
    for ax, (source, target) in zip(axes, source_target_list):
        #ax.set_title('{0} $\Rightarrow$ {1}'.format(
        #    anon_building_dict[source], anon_building_dict[target]))
        ax.text(0.45, 0.2, '{0} $\Rightarrow$ {1}'.format(
            anon_building_dict[source], anon_building_dict[target]),
            fontsize=11,
            transform=ax.transAxes)

    for i in range(1,len(source_target_list)):
        axes[i].set_yticklabels([])
        axes[i].set_ylabel('')

    ax = axes[0]
    handles, labels = ax.get_legend_handles_labels()
    legend_order = [0,1,2,3,4,5]
    new_handles = [handles[i] for i in legend_order]
    new_labels = [labels[i] for i in legend_order]
    ax.legend(new_handles, new_labels, bbox_to_anchor=(0.15,0.96), ncol=3, frameon=False)
    plt.text(0, 1.2, 'Accuracy: \nMacro $F_1$: ', ha='center', va='center',
            transform=ax.transAxes)
    fig.text(0.5, -0.1, '# of Target Building Samples', ha='center', 
            alpha=0)

    for i, ax in enumerate(axes):
        if i != 0:
            ax.set_xlabel('')

    fig.set_size_inches(4.4,1.5)
    save_fig(fig, 'figs/entity_iter.pdf')
    subprocess.call('./send_figures')

def entity_result_deprecated():
    source_target_list = [('ebu3b', 'ap_m')]#, ('ap_m', 'ebu3b')]
    n_list_list = [[(0,5), (0,50), (0,100), (0,150), (0,200)],
                   [(200,5), (200,50), (200,100), (0,150), (200,200)]]
    ts_flag = False
    eda_flag = False
    default_query = {
        'metadata.label_type': 'label',
        'metadata.token_type': 'justseparate',
        'metadata.use_cluster_flag': True,
        'metadata.building_list' : [],
        'metadata.source_sample_num_list': [],
        'metadata.target_building': '',
        'metadata.ts_flag': ts_flag,
        'metadata.eda_flag': eda_flag,
        'metadata.use_brick_flag': True
    }
    query_list = [deepcopy(default_query),\
                 deepcopy(default_query),\
                 deepcopy(default_query)]
    query_list[0]['metadata.use_brick_flag'] = False
    query_list[0]['metadata.negative_flag'] = False
    query_list[1]['metadata.use_brick_flag'] = False
    query_list[1]['metadata.negative_flag'] = True
    query_list[2]['metadata.use_brick_flag'] = True 
    query_list[2]['metadata.negative_flag'] = True
    char_precs_list = list()
    phrase_f1s_list = list()
    fig, axes = plt.subplots(1, 3)
#axes = [ax]
    fig.set_size_inches(8,5)
    #fig, axes = plt.subplots(1,len(n_list_list))

    for ax, (source, target) in zip(axes, source_target_list):
        for query in query_list:
            for n_list in n_list_list:
                target_n_list = [ns[1] for ns in n_list]
                subset_accuracy_list = list()
                accuracy_list = list()
                hierarchy_accuracy_list = list()
                weighted_f1_list = list()
                macro_f1_list = list()

                for (n_s, n_t) in n_list:
                    if n_s == 0:
                        building_list = [target]
                        source_sample_num_list = [n_t]
                    elif n_t == 0:
                        building_list = [source]
                        source_sample_num_list = [n_s]
                    else:
                        building_list = [source, target]
                        source_sample_num_list = [n_s, n_t]
                    query['metadata.building_list'] = building_list
                    query['metadata.source_sample_num_list'] = \
                            source_sample_num_list
                    query['metadata.target_building'] = target

                    result = get_entity_results(query)
                    try:
                        assert result
                    except:
                        print(n_t)
                        pdb.set_trace()
                        result = get_entity_results(query)
                    #point_precs = result['point_precision_history'][-1]
                    #point_recall = result['point_recall'][-1]
                    subset_accuracy_list.append(result['subset_accuracy_history'][-1] * 100)
                    accuracy_list.append(result['accuracy_history'][-1] * 100)
                    hierarchy_accuracy_list.append(result['hierarchy_accuracy_history'][-1] * 100)
                    weighted_f1_list.append(result['weighted_f1_history'][-1] * 100)
                    macro_f1_list.append(result['macro_f1_history'][-1] * 100)

                xs = target_n_list
                ys = [hierarchy_accuracy_list, accuracy_list, macro_f1_list]
                xlabel = '# of Target Building Samples'
                ylabel = 'Score (%)'
                xtick = target_n_list
                xtick_labels = [str(n) for n in target_n_list]
                ytick = range(0,102,10)
                ytick_labels = [str(n) for n in ytick]
                ylim = (ytick[0]-1, ytick[-1]+2)
                legends = [
                    '{0}, SA:{1}'\
                    .format(n_s, query['metadata.use_brick_flag']),
                    '{0}, SA:{1}'\
                    .format(n_s, query['metadata.use_brick_flag']),
                    '{0}, SA:{1}'\
                    .format(n_s, query['metadata.use_brick_flag'])
                          ]
                title = None
                plotter.plot_multiple_2dline(xs, ys, xlabel, ylabel, xtick,\
                                 xtick_labels, ytick, ytick_labels, title, ax, fig, \
                                 ylim, legends)
                #plotter.plot_multiple_2dline(xs, [ys[1]], xlabel, ylabel, xtick,\
                #                 xtick_labels, ytick, ytick_labels, title, axes[1], fig, \
                #                 ylim, [legends[1]])
                #plotter.plot_multiple_2dline(xs, [ys[2]], xlabel, ylabel, xtick,\
                #                 xtick_labels, ytick, ytick_labels, title, axes[2], fig, \
                #                 ylim, [legends[2]])
                if not (query['metadata.negative_flag'] and
                        query['metadata.use_brick_flag']):
                    break
    axes[0].set_title('Hierarchical Accuracy')
    axes[1].set_title('Accuracy')
    axes[2].set_title('Macro F1')
    suptitle = 'Multi Label (TagSets) Classification with a Source building.'
    fig.suptitle(suptitle)
    save_fig(fig, 'figs/entity.pdf')

def crf_entity_result():
    building_sets = [('ebu3b', 'ap_m'), ('ap_m', 'bml'),
                 ('ebu3b', 'ghc'), ('ghc', 'ebu3b'), ('ebu3b', 'bml', 'ap_m')] ### TODO TODO: this should be changed to use ebu3b,ap_m -> bml
    fig, axes = plt.subplots(1, len(building_sets))
    with open('result/baseline.json', 'r') as fp:
        baseline_results = json.load(fp)

    cs = ['firebrick', 'deepskyblue']
    plot_list = list()

    for i, (ax, buildings) in enumerate(zip(axes, building_sets)):
        print(i)
        # Baseline
        result = baseline_results[str(buildings)]
        init_ns = result['ns']
        sample_numbers = result['sample_numbers']
        avg_acc = result['avg_acc']
        std_acc = result['std_acc']
        avg_mf1 = result['avg_mf1']
        std_mf1 = result['std_mf1']
        xlabel = '# Target Building Samples'
        ys = [avg_acc, avg_mf1]
        x = sample_numbers
        xtick = sample_numbers
        xtick_labels = [str(no) for no in sample_numbers]
        ytick = list(range(0, 105, 20))
        ytick_labels = [str(no) for no in ytick]
        ylabel = 'Score (%)'
        ylabel_flag = False
        ylim = (-2, 105)
        xlim = (10, 205)
        linestyles = [':', ':']
        if i == 2:
            data_labels = ['Baseline Accuracy', 'Baseline Macro $F_1$']
        else:
            data_labels = None
        title = anon_building_dict[buildings[0]]
        for building in  buildings[1:-1]:
            title += ',{0}'.format(anon_building_dict[building])
        title += '$\\Rightarrow${0}'.format(anon_building_dict[buildings[-1]])
        lw = 1.2
        _, plot = plotter.plot_multiple_2dline(x, ys, xlabel, ylabel, xtick,
                             xtick_labels, ytick, ytick_labels, title,
                             ax, fig, ylim, xlim, data_labels, 0, linestyles,
                                               cs, lw)
        # scrabble
        if ''.join(buildings) == 'ebu3bbmlap_m':
            srcids_offset = 400
        else:
            srcids_offset = 200

        try:
            with open('result/crf_entity_iter_{0}.json'.format(''.join(buildings)),
                      'r') as fp:
                result = json.load(fp)[0]
        except:
            pdb.set_trace()
            continue
        zerofile = 'result/crf_entity_iter_{0}_zero.json'.format(''.join(buildings))
        if os.path.isfile(zerofile):
            with open(zerofile, 'r') as fp:
                zero_result = json.load(fp)[0]
            x_zero = [0]
            acc_zero = [zero_result['result']['entity'][0]['accuracy'] * 100]
            mf1_zero =  [zero_result['result']['entity'][0]['macro_f1'] * 100]
        else:
            x_zero = []
            acc_zero = []
            mf1_zero = []

        x = x_zero + [len(learning_srcids) - srcids_offset for learning_srcids in
             result['learning_srcids_history'][:-1]]
        accuracy= acc_zero + [res['accuracy'] * 100 for res in result['result']['entity']]
        mf1s = mf1_zero + [res['macro_f1'] * 100 for res in result['result']['entity']]
        ys = [accuracy, mf1s]
        linestyles = ['-', '-']
        if i == 2:
            data_labels = ['Scrabble Accuracy', 'Scrabble Macro $F_1$']
        else:
            data_labels = None
        _, plot = plotter.plot_multiple_2dline(x, ys, xlabel, ylabel, xtick,
                             xtick_labels, ytick, ytick_labels, title,
                             ax, fig, ylim, xlim, data_labels, 0, linestyles,
                                               cs, lw)
        if i == 2:
            ax.legend(bbox_to_anchor=(3.2, 1.45), ncol=4, frameon=False)
        plot_list.append(plot)


    fig.set_size_inches(9, 1.5)
    for ax in axes:
        ax.grid(True)
    for i in range(1,len(building_sets)):
        axes[i].set_yticklabels([])
        axes[i].set_ylabel('')
    for i in range(0,len(building_sets)):
        if i != 2:
            axes[i].set_xlabel('')

    #legends_list = ['Baseline A', 'Baseline MF']
    #axes[2].legend(loc='best', legends_list)


    save_fig(fig, 'figs/crf_entity.pdf')
    subprocess.call('./send_figures')

def crf_result():
    source_target_list = [('ebu3b', 'bml'), ('ghc', 'ebu3b')]
    n_list_list = [[(1000, 0), (1000,5), (1000,20), (1000,50), (1000,100), (1000, 150), (1000,200)],
                   [(200, 0), (200,5), (200,20), (200,50), (200,100), (200, 150), (200,200)],
                   [(0,5), (0,20), (0,50), (0,100), (0,150), (0,200)]]
    char_precs_list = list()
    phrase_f1s_list = list()
#fig, ax = plt.subplots(1, 1)
    fig, axes = plt.subplots(1,len(source_target_list))
    if isinstance(axes, Axes):
        axes = [axes]
    fig.set_size_inches(4, 1.5)
    cs = ['firebrick', 'deepskyblue']

    for i, (ax, (source, target)) in enumerate(zip(axes, source_target_list)):
        linestyles = ['--', '-.', '-']
        plot_list = list()
        legends_list = list()
        for n_list in n_list_list:
            target_n_list = [ns[1] for ns in n_list]
            phrase_f1s = list()
            char_macro_f1s = list()
            phrase_macro_f1s = list()
#pess_phrase_f1s = list()
            char_precs = list()
            for (n_s, n_t) in n_list:
                if n_s == 0:
                    building_list = [target]
                    source_sample_num_list = [n_t]
                elif n_t == 0:
                    building_list = [source]
                    source_sample_num_list = [n_s]
                else:
                    building_list = [source, target]
                    source_sample_num_list = [n_s, n_t]
                result_query = {
                    'label_type': 'label',
                    'token_type': 'justseparate',
                    'use_cluster_flag': True,
                    'building_list': building_list,
                    'source_sample_num_list': source_sample_num_list,

                    'target_building': target
                }
                result = get_crf_results(result_query)
                try:
                    assert result
                except:
                    print(n_t)
                    pdb.set_trace()
                    continue
                    result = get_crf_results(result_query)
                char_prec = result['char_precision'] * 100
                char_precs.append(char_prec)
                phrase_recall = result['phrase_recall'] * 100
                phrase_prec = result['phrase_precision'] * 100
                phrase_f1 = 2* phrase_prec  * phrase_recall \
                                / (phrase_prec + phrase_recall)
                phrase_f1s.append(phrase_f1)
                char_macro_f1s.append(result['char_macro_f1'] * 100)
                phrase_macro_f1s.append(result['phrase_macro_f1'] * 100)
            xs = target_n_list
            ys = [phrase_f1s, phrase_macro_f1s]
            #ys = [char_precs, phrase_f1s, char_macro_f1s, phrase_macro_f1s]
            #xlabel = '# of Target Building Samples'
            xlabel = None
            ylabel = 'Score (%)'
            xtick = list(range(0, 205, 40))
            #xtick = [0] + [5] + xtick[1:]
            xtick_labels = [str(n) for n in xtick]
            ytick = range(0,101,20)
            ytick_labels = [str(n) for n in ytick]
            xlim = (xtick[0]-2, xtick[-1]+5)
            ylim = (ytick[0]-2, ytick[-1]+5)
            if i == 0:
                legends = [#'#S:{0}, Char Prec'.format(n_s),
                    '#$B_S$:{0}'.format(n_s),
#'#S:{0}, Char MF1'.format(n_s),
                    '#$B_S$:{0}'.format(n_s),
                ]
            else:
                legends = None
#legends_list += legends
            title = None
            _, plots = plotter.plot_multiple_2dline(xs, ys, xlabel, ylabel, xtick,\
                             xtick_labels, ytick, ytick_labels, title, ax, fig, \
                             ylim, xlim, legends, xtickRotate=0, \
                             linestyles=[linestyles.pop()]*len(ys), cs=cs)
            plot_list += plots

#fig.legend(plot_list, legends_list, 'upper center', ncol=3
#            , bbox_to_anchor=(0.5,1.3),frameon=False)
    axes[0].legend(bbox_to_anchor=(0.15, 0.96), ncol=3, frameon=False)
    for ax in axes:
        ax.grid(True)
    axes[1].set_yticklabels([])
    axes[1].set_ylabel('')
    plt.text(0, 1.16, '$F_1$: \nMacro $F_1$: ', va='center', ha='center', 
            transform=axes[0].transAxes)
    fig.text(0.5, -0.1, '# of Target Building Samples', ha='center')

    save_fig(fig, 'figs/crf.pdf')
    subprocess.call('./send_figures')

def etc_result():
    buildings = ['ebu3b', 'bml', 'ap_m', 'ghc']
    tagsets_dict = dict()
    tags_dict = dict()
    tagset_numbers = []
    avg_required_tags = []
    tagset_type_numbers = []
    tags_numbers = []
    tags_type_numbers = []
    avg_tagsets = []
    median_occ_numbers = []
    avg_tags = []
    avg_tokens = []#TODO: Need to use words other than tokens
    avg_unfound_tags = []#TODO: Need to use words other than tokens
    once_numbers = []
    top20_numbers = []
    std_tags = []
    std_tagsets = []

    ignore_tagsets = ['leftidentifier', 'rightidentifier', 'none', 'unknown']
    total_tagsets = list(set(point_tagsets + location_tagsets + equip_tagsets))
    total_tags = list(set(reduce(adder, map(splitter, total_tagsets))))

    for building in buildings:
        with open('metadata/{0}_ground_truth.json'\
                  .format(building), 'r') as fp:
            truth_dict = json.load(fp)
        with open('metadata/{0}_label_dict_justseparate.json'.\
                  format(building), 'r') as fp:
            label_dict = json.load(fp)
        with open('metadata/{0}_sentence_dict_justseparate.json'\
                  .format(building), 'r') as fp:
            sentence_dict = json.load(fp)
        new_label_dict = dict()
        for srcid, labels in label_dict.items():
            new_label_dict[srcid] = list(reduce(adder, [label.split('_')
                                                   for label in labels if label
                                                   not in ignore_tagsets]))
        label_dict = new_label_dict
        srcids = list(label_dict.keys())
        label_dict = OrderedDict([(srcid, label_dict[srcid])
                                 for srcid in srcids])
        truth_dict = OrderedDict([(srcid, truth_dict[srcid])
                                 for srcid in srcids])
        sentence_dict = OrderedDict([(srcid, sentence_dict[srcid])
                                    for srcid in srcids])

        tagsets = [tagset for tagset in
                   list(reduce(adder, truth_dict.values()))
                   if tagset not in ignore_tagsets]
        def tagerize(tagsets):
            return list(set(reduce(adder, map(splitter, tagsets))))
        required_tags = list(map(tagerize, truth_dict.values()))
        tags = list(reduce(adder, map(splitter, tagsets)))
        tagsets_counter = Counter(tagsets)
        tagsets_dict[building] = Counter(tagsets)
        tagset_numbers.append(len(tagsets))
        tags_numbers.append(len(tags))
        tagset_type_numbers.append(len(set(tagsets)))
        tags_type_numbers.append(len(set(tags)))
        tokens_list = [[token for token in tokens
                        if re.match('[a-zA-Z]+', token)]
                       for tokens in sentence_dict.values()]
        unfound_tags_list = list()
        for srcid, tagsets in truth_dict.items():
            unfound_tags = set()
            for tagset in tagsets:
                for tag in tagset.split('_'):
                    if tag not in label_dict[srcid]:
                        unfound_tags.add(tag)
            unfound_tags_list.append(unfound_tags)
        avg_tokens.append(np.mean(list(map(lengther, tokens_list))))
        avg_tags.append(np.mean(list(map(lengther, map(set,label_dict.values())))))
        std_tags.append(np.std(list(map(lengther, map(set,label_dict.values())))))
        avg_tagsets.append(np.mean(list(map(lengther, truth_dict.values()))))
        std_tagsets.append(np.std(list(map(lengther, truth_dict.values()))))
        avg_required_tags.append(np.mean(list(map(lengther, required_tags))))
        avg_unfound_tags.append(np.mean(list(map(lengther, unfound_tags_list))))
        once_occurring_tagsets = [tagset for tagset, cnt
                                 in tagsets_counter.items() if cnt==1]

        once_numbers.append(len(once_occurring_tagsets))
        top20_numbers.append(np.sum(sorted(tagsets_counter.values(),
                                           reverse=True)[0:20]))
        median_occ_numbers.append(np.median(list(tagsets_counter.values())))

    tags_cnt = 0
    for tagset in total_tagsets:
        tags_cnt += len(splitter(tagset))
    avg_len_tagset = tags_cnt / len(total_tagsets)
    print('tot tags :', tagset_numbers)
    print('tot tagsets:', tags_numbers)
    print('avg len tagset:', avg_len_tagset)
    print('avg tokens:', avg_tokens)
    print('avg tags :', avg_tags)
    print('std tags :', std_tags)
    print('avg tagsets:', avg_tagsets)
    print('std tagsets:', std_tagsets)
    print('avg required tags:', avg_required_tags)
    print('avg unfound tags:', avg_unfound_tags)
    print('tot tagset:', tagset_type_numbers)
    print('tot tags:', tags_type_numbers)
    print('top20 explains: ', top20_numbers)
    print('num once occur: ', once_numbers)
    print('median occs: ', median_occ_numbers)
    print('brick tagsets: ', len(total_tagsets))
    print('brick tags: ', len(total_tags))


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
                                 'init', 'result'],
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
    parser.add_argument('-wk',
                        type=int,
                        help='Number of workers for high level MP',
                        dest='worker_num',
                        default=2)
    parser.add_argument('-nj',
                        type=int,
                        help='Number of processes for multiprocessing',
                        dest='n_jobs',
                        default=4)
    parser.add_argument('-ct',
                        type=str,
                        help='Tagset classifier type. one of RandomForest, \
                              StructuredCC.',
                        dest='tagset_classifier_type',
                        default='StructuredCC')
    parser.add_argument('-ts',
                        type='bool',
                        help='Flag to use time series features too',
                        dest='ts_flag',
                        default=False)
    parser.add_argument('-neg',
                        type='bool',
                        help='Negative Samples augmentation',
                        dest='negative_flag',
                        default=True)
    parser.add_argument('-exp', 
                        type=str,
                        help='type of experiments for result output',
                        dest = 'exp_type')
    parser.add_argument('-post', 
                        type=str,
                        help='postfix of result filename',
                        default='0',
                        dest = 'postfix')

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
                                         args.ts_flag,
                                         args.negative_flag,
                                         args.n_jobs
                                        )
        elif args.avgnum>1:
            entity_recognition_from_ground_truth_get_avg(args.avgnum,
                building_list=args.source_building_list,
                source_sample_num_list=args.sample_num_list,
                target_building=args.target_building,
                token_type='justseparate',
                label_type=args.label_type,
                use_cluster_flag=args.use_cluster_flag,
                use_brick_flag=args.use_brick_flag,
                eda_flag=args.eda_flag,
                ts_flag=args.ts_flag,
                negative_flag=args.negative_flag,
                n_jobs=args.n_jobs,
                worker_num=args.worker_num)
    elif args.prog == 'crf_entity':
        params = (args.source_building_list,
                  args.sample_num_list,
                  args.target_building,
                  'justseparate',
                  args.label_type,
                  args.use_cluster_flag,
                  args.use_brick_flag,
                  args.eda_flag,
                  args.negative_flag,
                  args.debug_flag,
                  args.n_jobs,
                  args.ts_flag)
        crf_entity_recognition_iteration(args.iter_num, args.postfix, *params)

        """
        entity_recognition_from_crf(\
                building_list=args.source_building_list,\
                source_sample_num_list=args.sample_num_list,\
                target_building=args.target_building,\
                token_type='justseparate',\
                label_type=args.label_type,\
                use_cluster_flag=args.use_cluster_flag,\
                use_brick_flag=args.use_brick_flag,\
                eda_flag=args.eda_flag,
                debug_flag=args.debug_flag,
                n_jobs=args.n_jobs)
        """
    elif args.prog == 'result':
        assert args.exp_type in ['crf', 'entity', 'crf_entity', 'entity_iter',
                                 'etc', 'entity_ts']
        if args.exp_type == 'crf':
            crf_result()
        elif args.exp_type == 'entity':
            entity_result()
        elif args.exp_type == 'crf_entity':
            crf_entity_result()
        elif args.exp_type == 'entity_iter':
            entity_iter_result()
        elif args.exp_type == 'entity_ts':
            entity_ts_result()
        elif args.exp_type == 'etc':
            etc_result()

    elif args.prog == 'init':
        init()
    else:
        #print('Either learn or predict should be provided')
        assert(False)
