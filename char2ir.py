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
from common import *


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
        #if i<len(sentence)-1:
        #    features['+1:word.lower='+sentence[i+1].lower()] = 1.0
        #else:
        #    features['EOS'] = 1.0
        sentenceFeatures.append(features)
    return sentenceFeatures


def learn_crf_model(building_list,
                    source_sample_num_list,
                    use_cluster_flag=False,
                    use_brick_flag=False,
                    prev_step_data={
                        'learning_srcids':[], #TODO: Make sure it includes newly added samples.
                        'iter_cnt':0
                    },
                    oversample_flag=False
                   ):
    """
    Input: 
        building_list: building names for learning
        source_sample_num_list: number of samples to pick from each building.
                                This has to be aligned with building_list
        use_cluster_flag: whether to select samples heterogeneously or randomly
        use_brick_flag: whether to include brick tags in the sample.
        prev_step_data: Previous step information for iteration.
        oversample_flag: duplicate examples to overfit the model. Not used.
    """

    assert(isinstance(building_list, list))
    assert(isinstance(source_sample_num_list, list))
    assert(len(building_list)==len(source_sample_num_list))

    data_feature_flag = False # Determine to use data features here or not.
                              # Currently not used at this stage.
    sample_dict = dict()

    # It does training and testing on the same building but different points.

    crf_model_file = 'temp/{0}.crfsuite'.format(gen_uuid())
    log_filename = 'logs/training_{0}_{1}_{2}.log'\
            .format(building_list[0], source_sample_num_list[0], \
                    'clustered' if use_cluster_flag else 'unclustered')
    logging.basicConfig(filename=log_filename, 
                        level=logging.DEBUG,
                        format='%(asctime)s %(message)s')
    logging.info('{0}th Start learning CRF model'.format(\
                                                prev_step_data['iter_cnt']))

    ### TRAINING PHASE ###
    # Init CRFSuite
    trainer = pycrfsuite.Trainer(verbose=False, algorithm='pa')
    # algorithm: {'lbfgs', 'l2sgd', 'ap', 'pa', 'arow'}
    trainer.set_params({'feature.possible_states': True,
                        'feature.possible_transitions': True})
    #data_available_buildings = []

    # Select samples
    learning_srcids = list()

    for building, source_sample_num in zip(building_list, 
                                           source_sample_num_list):
        #  Load raw sentences of a building
        with open("metadata/%s_char_sentence_dict.json" % (building), "r") as fp:
            sentence_dict = json.load(fp)
        sentence_dict = dict((srcid, [char for char in sentence]) for (srcid, sentence) in sentence_dict.items())

        # Load character label mappings.
        with open('metadata/{0}_char_label_dict.json'.format(building), 'r') as fp:
            label_dict = json.load(fp)

        # Select learning samples.
        if prev_step_data['learning_srcids']:
            sample_srcid_list = [srcid for srcid in sentence_dict.keys() \
                                 if srcid in prev_step_data['learning_srcids']] # TODO: This seems to be "NOT if srcid in~~" Validate it.
        else:
            sample_srcid_list = select_random_samples(building, \
                                                      label_dict.keys(), \
                                                      source_sample_num, \
                                                      use_cluster_flag)
        learning_srcids += sample_srcid_list
        
        if oversample_flag:
            sample_srcid_list = sample_srcid_list * \
                                floor(1000 / len(sample_srcid_list))

        # Add samples to the trainer.
        for srcid in sample_srcid_list:
            sentence = list(map(itemgetter(0), label_dict[srcid]))
            labels = list(map(itemgetter(1), label_dict[srcid]))
            trainer.append(pycrfsuite.ItemSequence(
                calc_features(sentence, None)), labels)
        sample_dict[building] = list(sample_srcid_list)
    if prev_step_data.get('learning_srcids_history'):
        assert set(prev_step_data['learning_srcids_history'][-1]) == \
                set(learning_srcids)


    # Add Brick tags to the trainer.
    if use_brick_flag:
        with open('metadata/brick_tags_labels.json', 'r') as fp:
            tag_label_list = json.load(fp)
        for tag_labels in tag_label_list:
            # Append meaningless characters before and after the tag
            # to make it separate from dependencies.
            # But comment them out to check if it works.
            #char_tags = [' '] + list(map(itemgetter(0), tag_labels)) + [' ']
            char_tags = list(map(itemgetter(0), tag_labels))
            #char_labels = ['O'] + list(map(itemgetter(1), tag_labels)) + ['O']
            char_labels = list(map(itemgetter(1), tag_labels))
            trainer.append(pycrfsuite.ItemSequence(
                calc_features(char_tags, None)), char_labels)


    # Train and store the model file
    trainer.train(crf_model_file)
    with open(crf_model_file, 'rb') as fp:
        model_bin = fp.read()
    model = {
        'source_list': sample_dict,
        'gen_time': arrow.get().datetime, #TODO: change this to 'date'
        'use_cluster_flag': use_cluster_flag,
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
             use_cluster_flag=False,
             use_brick_flag=False,
             learning_srcids=[]
            ):
    """
    similar to learn_crf_model
    """
    assert len(building_list) == len(source_sample_num_list)

    source_building_names = '_'.join(building_list)

    # Retrieve model from mongodb
    model_query = {'$and':[]}
    model_metadata = {
        'use_cluster_flag': use_cluster_flag,
        'source_building_count': len(building_list),
    }
    model_query['$and'].append(model_metadata)
    model_query['$and'].append({'source_building_count':len(building_list)})

    # Store the model configuration for results metadata
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
    if learning_srcids:
        learning_srcids = sorted(learning_srcids)
        model_query = {'learning_srcids': learning_srcids}
    try:
        # Retrieve the most recent model matching the query
        model = get_model(model_query)
    except:
        pdb.set_trace()
    result_metadata['source_list'] = model['source_list']

    if not learning_srcids:
        learning_srcids = sorted(list(reduce(adder, model['source_list'].values())))
    assert sorted(learning_srcids) == sorted(list(reduce(adder, model['source_list'].values())))

    result_metadata['learning_srcids'] = learning_srcids

    # Store model file read from MongoDB to read from tagger.
    crf_model_file = 'temp/{0}.crfsuite'.format(gen_uuid())
    with open(crf_model_file, 'wb') as fp:
        fp.write(model['model_binary'])

    # Init resulter (storing/interpreting results
    resulter = Resulter(spec=result_metadata)
    log_filename = 'logs/test_{0}_{1}_{2}_{3}.log'\
            .format(source_building_names, 
                    target_building,
                    source_sample_num, 
                    'clustered' if use_cluster_flag else 'unclustered')
    logging.basicConfig(filename=log_filename, 
                        level=logging.DEBUG,
                        format='%(asctime)s %(message)s')
    logging.info("Started!!!")

    data_available_buildings = []

    with open('metadata/{0}_char_label_dict.json'\
                .format(target_building), 'r') as fp:
        target_label_dict = json.load(fp)
    with open('metadata/{0}_char_sentence_dict.json'\
                .format(target_building), 'r') as fp:
        sentence_dict = json.load(fp)

    # Select only sentences with ground-truth for testing.
    sentence_dict = dict((srcid, sentence) 
                         for srcid, sentence 
                         in sentence_dict.items() 
                         if target_label_dict.get(srcid))

    # Init tagger
    tagger = pycrfsuite.Tagger()
    tagger.open(crf_model_file)

    # Tagging sentences with tagger
    predicted_dict = dict()
    score_dict = dict()
    for srcid, sentence in sentence_dict.items():
        predicted = tagger.tag(calc_features(sentence))
        predicted_dict[srcid] = predicted
        score_dict[srcid] = tagger.probability(predicted)

    # Analysis of the result.
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

    result_file = 'result/test_result_{0}_{1}_{2}_{3}.json'\
                    .format(source_building_names,
                            target_building,
                            source_sample_num,
                            'clustered' if use_cluster_flag else 'unclustered')
    summary_file = 'result/test_summary_{0}_{1}_{2}_{3}.json'\
                    .format(source_building_names,
                            target_building,
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

    error_plot_file = 'figs/error_plotting_{0}_{1}_{2}_{3}.pdf'\
                    .format(source_building_names, 
                            target_building,
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
