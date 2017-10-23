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

from distance import levenshtein as editdistance

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
from crfsharp import CRFSharp
from common import *


crfsharp_other_postfixes = ['.alpha', '.feature', '.feature.raw_text']
thread_num = 28
crfsharp_maxiter = 20
#logger = get_logger()


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
                    crftype='crfsuite',
                    step_data={
                        'next_learning_srcids':[], #TODO: Make sure it includes newly added samples.
                        'iter_num':0,
                        'model_uuid': None
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
        step_data: Previous step information for iteration.
        oversample_flag: duplicate examples to overfit the model. Not used.
    """
    begin_time = arrow.get()
    assert(isinstance(building_list, list))
    assert(isinstance(source_sample_num_list, list))
    assert(len(building_list)==len(source_sample_num_list))

    data_feature_flag = False # Determine to use data features here or not.
                              # Currently not used at this stage.
    sample_dict = dict()

    # It does training and testing on the same building but different points.

    log_filename = os.path.dirname(os.path.abspath(__file__)) + \
            '/logs/training_{0}_{1}_{2}.log'\
            .format(building_list[0], source_sample_num_list[0], \
                    'clustered' if use_cluster_flag else 'unclustered')
    
    #logger = get_logger(log_filename)
    logger = logging.getLogger()
    logger.info('{0}th Start learning CRF model'.format(step_data['iter_num']))

    ### TRAINING PHASE ###
    # Select samples
    learning_srcids = list()
    sentence_dict = dict()
    label_dict = dict()

    for building, source_sample_num in zip(building_list, 
                                           source_sample_num_list):
        #  Load raw sentences of a building
        with open("metadata/%s_char_sentence_dict.json" % (building), "r") \
                as fp:
            one_sentence_dict = json.load(fp)
        one_sentence_dict = dict((srcid, [char for char in sentence]) for \
                (srcid, sentence) in one_sentence_dict.items())

        # Load character label mappings.
        with open('metadata/{0}_char_label_dict.json'.format(building), 'r')\
                as fp:
            one_label_dict = json.load(fp)

        # Select learning samples.
        # Learning samples can be chosen from the previous stage.
        # Or randomly selected.
        if step_data['next_learning_srcids']:
            sample_srcid_list = [srcid for srcid in one_sentence_dict.keys() \
                                 if srcid in step_data['next_learning_srcids']] 
                    # TODO: This seems to be "NOT if srcid in~~" Validate it.
        else:
            sample_srcid_list = select_random_samples(building, \
                                                      one_label_dict.keys(), \
                                                      source_sample_num, \
                                                      use_cluster_flag)
        learning_srcids += sample_srcid_list
        
        if oversample_flag:
            sample_srcid_list = sample_srcid_list * \
                                floor(1000 / len(sample_srcid_list))
        sample_dict[building] = list(sample_srcid_list) # For debugging

        # Add samples to the trainer.
        """
        for srcid in sample_srcid_list:
            sentence = list(map(itemgetter(0), one_label_dict[srcid]))
            labels = list(map(itemgetter(1), one_label_dict[srcid]))
            trainer.append(pycrfsuite.ItemSequence(
                calc_features(sentence, None)), labels)
        """
        for srcid in sample_srcid_list:
            sentence_dict[srcid] = list(map(itemgetter(0), one_label_dict[srcid]))
            label_dict[srcid] = list(map(itemgetter(1), one_label_dict[srcid]))
        ## TODO: Add info extraction from target building if necessary?


    if step_data.get('learning_srcids_history'):
        # learning_srcids_history's last item is 
        # actually current learning srcids.
        #assert set(step_data['learning_srcids_history'][-1]) == \
        #        set(learning_srcids)
        assert set(step_data['learning_srcids']) == set(learning_srcids)


    # Add Brick tags to the trainer. 
    # Commonly disabled. Accuracy gets worse. Keep it for history
    brick_sentence_dict = dict()
    brick_label_dict = dict()
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
            brick_sentence_dict[''.join(char_tags)] = char_tags + ['NEWLINE']
            brick_label_dict[''.join(char_tags)] = char_labels + ['O']
            #trainer.append(pycrfsuite.ItemSequence(
            #    calc_features(char_tags, None)), char_labels)
    brick_srcids = list(brick_sentence_dict.keys())
    
    # Add samples to the trainer
    if crftype == 'crfsharp':
        if step_data['model_uuid']:
            # If previous step contains model, retrain a model from it.
            # TODO: Validate retrain generates same tags
            #prev_model_uuid = step_data['model_uuid']
            #prev_model_filename = './temp/model/' + prev_model_uuid + \
            #                            '.crfsharp.model'
            #prev_model = get_model({'uuid': prev_model_uuid})
            #load_crfsharp_model_files(prev_model, prev_model_filename, crftype)
            prev_model_filename = None
            curr_learning_srcids = learning_srcids
        else:
            prev_model_filename = None
            curr_learning_srcids = learning_srcids
        trainer = CRFSharp(base_dir = './temp', \
                           template = './model/scrabble.template',
                           thread = thread_num,
                           maxiter = crfsharp_maxiter,
                           nbest = 1)
        sentences = [sentence_dict[srcid] for srcid in curr_learning_srcids] #+\
                        #[brick_sentence_dict[srcid] for srcid in brick_srcids]
        labels = [label_dict[srcid] for srcid in curr_learning_srcids] #+ \
                     #[brick_label_dict[srcid] for srcid in brick_srcids]
        if use_brick_flag and not prev_model_filename:
            prev_model_filename = 'model/brick_crfsharp.model'
        crf_model_file = trainer.encode(sentences, labels, prev_model_filename)
    elif crftype == 'crfsuite':
        # Init CRFSuite
        algo = 'ap'
        trainer = pycrfsuite.Trainer(verbose=False, algorithm=algo)
        if algo == 'ap':
            trainer.set('max_iterations', 200)

                         # algorithm: {'lbfgs', 'l2sgd', 'ap', 'pa', 'arow'}
        trainer.set_params({'feature.possible_states': True,
                            'feature.possible_transitions': True})
        for srcid in learning_srcids:
            sentence = sentence_dict[srcid]
            labels = label_dict[srcid]
            trainer.append(pycrfsuite.ItemSequence(
                calc_features(sentence, None)), labels)
        for srcid in brick_srcids:
            sentence = brick_sentence_dict[srcid]
            labels = brick_label_dict[srcid]
            trainer.append(pycrfsuite.ItemSequence(
                calc_features(sentence, None)), labels)
        crf_model_file = 'temp/{0}.{1}.model'.format(gen_uuid(), \
                                                     crftype)
        trainer.train(crf_model_file)
    else:
        raise Exception('Not existing crf package name: ' + crftype)
        

    # Train and store the model file
    with open(crf_model_file, 'rb') as fp:
        model_bin = fp.read()
    model_uuid = gen_uuid()
    model = {
        'source_list': sample_dict,
        'gen_time': arrow.get().datetime, #TODO: change this to 'date'
        'use_cluster_flag': use_cluster_flag,
        'model_binary': BsonBinary(model_bin),
        'source_building_count': len(building_list),
        'learning_srcids': sorted(learning_srcids),
        'uuid': model_uuid,
        'crftype': crftype
    }
    if crftype == 'crfsharp':
        for postfix in crfsharp_other_postfixes:
            with open(crf_model_file + postfix, 'rb') as fp:
                model_bin = fp.read()
                model['model_binary' + postfix.replace('.', '_')] = BsonBinary(model_bin)

    store_model(model)
    os.remove(crf_model_file)

    end_time = arrow.get()
    logger.info('CRF Learning took: ' + str(end_time - begin_time))
    logger.info('Finished!!!')
    return model_uuid

def crf_test(building_list,
             source_sample_num_list,
             target_building,
             use_cluster_flag=False,
             use_brick_flag=False,
             crftype='crfsuite',
             learning_srcids=[],
             model_uuid=None
            ):
    """
    Inputs are similar to learn_crf_model
    """
    assert len(building_list) == len(source_sample_num_list)

    source_building_names = '_'.join(building_list)

    # Define a query to retrieve model from mongodb
    if model_uuid:
        model_query = {'uuid': model_uuid}
    else: 
        model_query = {'$and':[]}
        model_query['$and'].append({'crftype': crftype})
        model_metadata = {
            'use_cluster_flag': use_cluster_flag,
            'source_building_count': len(building_list),
        }
        model_query['$and'].append(model_metadata)
        model_query['$and'].append({'source_building_count':len(building_list)})

        # Store the model configuration for results metadata
        for building, source_sample_num in \
                zip(building_list, source_sample_num_list):
            model_query['$and'].append(
                {'source_list.{0}'.format(building): {'$exists': True}})
            model_query['$and'].append({'$where': \
                                        'this.source_list.{0}.length=={1}'.\
                                        format(building, source_sample_num)})
            #result_metadata['source_cnt_list'].append([building, source_sample_num])
        if learning_srcids:
            learning_srcids = sorted(learning_srcids)
            model_query = {'learning_srcids': learning_srcids}

    try:
        # Retrieve the most recent model matching the query
        model = get_model(model_query)
        if model_uuid:
            model_metadata = model # TODO: Validate it.
    except:
        pdb.set_trace()
    result_metadata = deepcopy(model_metadata)
    result_metadata['source_cnt_list'] = []
    result_metadata['target_building'] = target_building
    result_metadata['source_list'] = model['source_list']
    for building, source_sample_num in \
            zip(building_list, source_sample_num_list):
        result_metadata['source_cnt_list'].append([building, source_sample_num])

    if not learning_srcids:
        learning_srcids = sorted(list(reduce(adder, model['source_list'].values())))
    assert set(learning_srcids) == set(list(reduce(adder, model['source_list'].values())))

    result_metadata['learning_srcids'] = learning_srcids

    # Init resulter (a module storing/interpreting results)
    resulter = Resulter(spec=result_metadata)
    log_filename = './logs/test_{0}_{1}_{2}_{3}.log'\
            .format(source_building_names, 
                    target_building,
                    source_sample_num, 
                    'clustered' if use_cluster_flag else 'unclustered')
    logger = logging.getLogger()
    #logger = get_logger(log_filename)
    logger.info("Started!!!")

    with open('metadata/{0}_char_label_dict.json'\
                .format(target_building), 'r') as fp:
        target_label_dict = json.load(fp)
    with open('metadata/{0}_char_sentence_dict.json'\
                .format(target_building), 'r') as fp:
        sentence_dict = json.load(fp)

    # Limit target number for testing
    target_limit = -1
    new_target_label_dict = dict()
    for srcid, target_label in list(target_label_dict.items())[0:target_limit]:
        new_target_label_dict[srcid] = target_label
    target_label_dict = new_target_label_dict

    # Select only sentences with ground-truth for testing.
    sentence_dict = dict((srcid, sentence) 
                         for srcid, sentence 
                         in sentence_dict.items()
                         if target_label_dict.get(srcid))
    
    # Store model file read from MongoDB to read from tagger.
    begin_time = arrow.get()
    predicted_dict, score_dict = predict_func(model, sentence_dict, crftype)
    end_time = arrow.get()
    logger.info('tagging took: ' + str(end_time - begin_time))

    # Analysis of the result.
    precisionOfTrainingDataset = 0
    totalWordCount = 0
    error_rate_dict = dict()

    # TODO: Validate working
    #for srcid, sentence_label in target_label_dict.items():
    for srcid, score in sorted(score_dict.items(), key=itemgetter(1)):
        sentence_label = target_label_dict[srcid]
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
        logger.info("=========== {0} ==== {1} ==================="\
                        .format(srcid, score_dict[srcid]))
        error_rate_dict[srcid] = sum([pair[0]=='X' for pair in printing_pairs])\
                                    /float(len(sentence))
        if 'X' in [pair[0] for pair in printing_pairs]:
            for (flag, word, predTag, origLabel) in printing_pairs:
                logger.info('{:5s} {:20s} {:20s} {:20s}'\
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

    logger.info("Finished!!!!!!")

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

    # Construct output data
    pred_phrase_dict = make_phrase_dict(sentence_dict, predicted_dict)
    step_data = {
        'learning_srcids': learning_srcids,
        'result': {
            'crf': resulter.get_summary()
        },
        'phrase_dict': pred_phrase_dict
    }
    for k, v in result_metadata.items():
        if k in ['_id', 'model_binary', 'gen_time', 'model_binary_alpha', 
                 'model_binary_feature', 'model_binary_feature_raw_text']:
            continue
        step_data[k] = v
    return step_data


def query_active_learning_samples(prev_learning_srcids,
                                  target_building,
                                  model_uuid,
                                  crftype='crfsuite',
                                  inc_num=10,
                                  query_strategy='confidence'):

    logger = logging.getLogger()
    # Load ground truth
    with open('metadata/{0}_char_label_dict.json'.format(target_building), 'r') as fp:
        label_dict = json.load(fp)
    with open('metadata/{0}_label_dict_justseparate.json'
                  .format(target_building), 
              'r') as fp:
        phrase_label_dict = json.load(fp)
    with open('metadata/{0}_char_sentence_dict.json'\
                .format(target_building), 'r') as fp:
        raw_sentence_dict = json.load(fp)
        sentence_dict = dict()
        for srcid in label_dict.keys():
            sentence_dict[srcid] = raw_sentence_dict[srcid]

    cand_srcids = [srcid for srcid in label_dict.keys() 
                   if srcid not in prev_learning_srcids]
    

    # Load model
    model = get_model({'uuid': model_uuid})


    # Predict CRF
    predicted_dict, score_dict = predict_func(model, sentence_dict, crftype)
        
        
    # Query new srcids for active learning
    cluster_dict = get_cluster_dict(target_building)
    if query_strategy == 'confidence':
        for srcid, score in score_dict.items():
            # Normalize with length
            score_dict[srcid] = np.log(score) / len(sentence_dict[srcid])
        sorted_scores = sorted(score_dict.items(), key=itemgetter(1))

        ### load word clusters not to select too similar samples.
        added_cids = []
        new_srcids = []
        new_srcid_cnt = 0
        for srcid, score in sorted_scores:
            if srcid not in prev_learning_srcids:
                the_cid = None
                for cid, cluster in cluster_dict.items():
                    if srcid in cluster:
                        the_cid = cid
                        break
                if the_cid in added_cids:
                    continue
                added_cids.append(the_cid)
                new_srcids.append(srcid)
                new_srcid_cnt += 1
                if new_srcid_cnt == inc_num:
                    break
    elif query_strategy == 'strict_confidence':
        for srcid, score in score_dict.items():
            # Normalize with length
            score_dict[srcid] = np.log(score) / len(sentence_dict[srcid])
        sorted_scores = sorted(score_dict.items(), key=itemgetter(1))

        ### load word clusters not to select too similar samples.
        added_cids = set()
        for srcid in prev_learning_srcids:
            for cid, cluster in cluster_dict.items():
                if srcid in cluster:
                    added_cids.add(cid)
        added_cids = list(added_cids)

        new_srcids = []
        new_srcid_cnt = 0
        for srcid, score in sorted_scores:
            if srcid not in prev_learning_srcids:
                the_cid = None
                for cid, cluster in cluster_dict.items():
                    if srcid in cluster:
                        the_cid = cid
                        break
                if the_cid in added_cids:
                    continue
                added_cids.append(the_cid)
                new_srcids.append(srcid)
                new_srcid_cnt += 1
                if new_srcid_cnt == inc_num:
                    break
        if new_srcid_cnt < inc_num:
            new_srcids += select_random_samples(target_building,
                                                cand_srcids,
                                                inc_num - new_srcid_cnt,
                                                use_cluster_flag=True,
                                                token_type='justseparate',
                                                reverse=True,
                                                cluster_dict=None,
                                                shuffle_flag=True
                                                )
    elif query_strategy == 'empty':
        new_srcids = []
    else:
        assert False

    next_learning_srcids = prev_learning_srcids + new_srcids * 3


    # Finish Condition
    fin_ratio = 0.9
    for srcid in new_srcids:
        pass


    ##########################
    # Debugging and Analysis #
    ##########################

    ### Check if known srcid covers corresponding clusters based on score
    high_score_cnt = 0
    low_score_cnt = 0
    high_score_srcids = []
    low_score_srcids = []
    avg_score = np.mean(list(score_dict.values()))
    for srcid in prev_learning_srcids:
        if srcid not in label_dict.keys():
            continue
        # Find corresponding cluster
        for cid, cluster in cluster_dict.items():
            if srcid in cluster:
                for sid in cluster:
                    if sid not in sentence_dict.keys():
                        continue
                    if score_dict[sid] > avg_score:
                        high_score_cnt += 1
                        high_score_srcids.append(sid)
                    else:
                        low_score_cnt += 1
                        low_score_srcids.append(sid)
                break
    if high_score_cnt == 0 and low_score_cnt == 0:
        high_score_rate = 0
    else:
        high_score_rate = high_score_cnt / (high_score_cnt + low_score_cnt)

    ### Check srcids in clusters related to newly added srcids
    subscore_dict = dict()
    in_func = lambda x,y: x in y
    for srcid in new_srcids:
        cid = find_keys(srcid, cluster_dict, in_func)[0]
        cluster = cluster_dict[cid]
        for sid in cluster:
            if sid in sentence_dict.keys() and sid != srcid:
                subscore_dict[sid] = score_dict[sid]
    print(subscore_dict)
        

    ### Check if known words are correctly inferred.
    pred_phrase_dict = make_phrase_dict(sentence_dict, predicted_dict)
    tot_score = 0
    cnt = 0
    splitter = lambda s: s.split('_')
    adder = lambda x,y: x + y
    for srcid in prev_learning_srcids:
        if srcid not in label_dict.keys():
            continue
        try:
            sentence = ''.join(sentence_dict[srcid])
        except:
            pdb.set_trace()
        pred_phrases = pred_phrase_dict[srcid]
        true_phrases = [phrase for phrase in phrase_label_dict[srcid] \
                        if phrase not in ['leftidentifier',
                                          'rightidentifier',
                                          'none']]
        true_phrases = reduce(adder, map(splitter, true_phrases), [])
        
        curr_score = editdistance(pred_phrases, true_phrases) \
                        / len(true_phrases)
        print('==================')
        print(sentence)
        print('pred', pred_phrases)
        print('true', true_phrases)
        print(curr_score)
        print('==================')
        tot_score += curr_score
        cnt += 1
    avg_score = tot_score / cnt
        
        
    ### Finding words known and unknown.
    with open('metadata/{0}_sentence_dict_justseparate.json'\
                  .format(target_building), 'r') as fp:
        word_sentence_dict = json.load(fp)
    srcids = list(word_sentence_dict.keys())
    norm_sentence_dict = dict()
    for srcid, sentence in word_sentence_dict.items():
        norm_sentence_dict[srcid] = list(map(replace_num_or_special, sentence))
    adder = lambda x,y: x + y
    discovered_words = Counter(reduce(adder, [sentence for srcid, sentence 
                                              in norm_sentence_dict.items() 
                                              if srcid 
                                              in prev_learning_srcids]))
    total_words = set(reduce(adder, norm_sentence_dict.values()))
    word_counter = Counter(reduce(adder, norm_sentence_dict.values()))
    noncovered_word_counter = dict()
    for word, cnt in word_counter.items():
        if word not in discovered_words.keys():
            noncovered_word_counter[word] = cnt

    ### Ratio of newly discovered words in newly added srcids
    new_words = list()
    for srcid in new_srcids:
        new_words += norm_sentence_dict[srcid]
    new_word_counter = Counter(new_words)
    newly_found_words = set([word for word in new_words
                             if word in noncovered_word_counter.keys()])
    if len(new_word_counter):
        logger.info('purity of new srcids ' + 
                    str(len(newly_found_words) / len(new_word_counter)))
    
    return next_learning_srcids
    

def char2ir_onestep(step_data,
                    building_list,
                    source_sample_num_list,
                    target_building,
                    use_cluster_flag=False,
                    use_brick_flag=False,
                    crftype='crfsuite',
                    inc_num=10,
                    query_strategy='confidence'
                    ):
    #                gen_next_step=True): TODO: Validate this
    step_data = deepcopy(step_data)
    begin_time = arrow.get()

    assert step_data.get('next_learning_srcids')
    learning_srcids = step_data['next_learning_srcids']
    

    # Learn Model
    model_uuid = learn_crf_model(building_list,
                    source_sample_num_list,
                    use_cluster_flag,
                    False, #use_brick_flag,
                    crftype,
                    step_data,
                    False # Oversample Flag,
                   )

    # Inference from model
    next_step_data = crf_test(building_list,
                   source_sample_num_list,
                   target_building,
                   use_cluster_flag,
                   False, #use_brick_flag,
                   crftype,
                   learning_srcids,
                   model_uuid=model_uuid
                  )

    # Ask query for active learning
    new_learning_srcids = query_active_learning_samples(
                              learning_srcids, 
                              target_building,
                              model_uuid,
                              crftype,
                              inc_num,
                              query_strategy)
    #next_step_data['iter_num'] = step_data['iter_num'] + 1
    next_step_data['learning_srcids'] = learning_srcids
    next_step_data['next_learning_srcids'] = new_learning_srcids
    next_step_data['added_learning_srcids'] = [srcid for srcid
                                               in new_learning_srcids
                                               if srcid not in 
                                               learning_srcids]
    next_step_data['model_uuid'] = model_uuid
    next_step_data['iter_num'] = step_data['iter_num']
    end_time = arrow.get()
    print('An interation took: ', end_time - begin_time)

    return next_step_data 

def load_crfsharp_model_files(model, filename, crftype):
    crf_model_file = filename
    with open(crf_model_file, 'wb') as fp:
        fp.write(model['model_binary'])
    if crftype == 'crfsharp':
        for postfix in crfsharp_other_postfixes:
            with open(crf_model_file + postfix, 'wb') as fp:
                fp.write(model['model_binary' + postfix.replace('.','_')])

def predict_func(model, sentence_dict, crftype):
    crf_model_file = 'temp/{0}.{1}.model'.format(gen_uuid(), crftype)
    """
    with open(crf_model_file, 'wb') as fp:
        fp.write(model['model_binary'])
    if crftype == 'crfsharp':
        for postfix in crfsharp_other_postfixes:
            with open(crf_model_file + postfix, 'wb') as fp:
                fp.write(model['model_binary' + postfix.replace('.','_')])
    """
    load_crfsharp_model_files(model, crf_model_file, crftype)

    predicted_dict = dict()
    score_dict = dict()
    begin_time = arrow.get()
    if crftype == 'crfsuite':
        # Init tagger
        tagger = pycrfsuite.Tagger()
        tagger.open(crf_model_file)

        # Tagging sentences with tagger
        for srcid, sentence in sentence_dict.items():
            predicted = tagger.tag(calc_features(sentence))
            predicted_dict[srcid] = predicted
            score_dict[srcid] = tagger.probability(predicted)
    elif crftype ==  'crfsharp':
        tagger = CRFSharp(base_dir = './temp', \
                           template = './model/scrabble.template',
                           thread = thread_num,
                           nbest = 1,
                           modelfile = crf_model_file,
                           maxiter=crfsharp_maxiter
                           )
        srcids = list(sentence_dict.keys())
        sentences = [sentence_dict[srcid] for srcid in srcids]
        res = tagger.decode(sentences, srcids)
        for srcid in srcids:
            best_cand = res[srcid]['cands'][0]
            predicted_dict[srcid] = best_cand['token_predict']
            score_dict[srcid] = best_cand['prop']
    return predicted_dict, score_dict
    

def char2ir_iteration(iter_num, custom_postfix='', *args):
    """
    args: 
        building_list,
        source_sample_num_list,
        target_building,
        use_cluster_flag=False,
        use_brick_flag=False,
        crftype='crfsuite'
        inc_num=10
        ):
    """
    logfilename = 'logs/char2ir_iter_{0}.log'.format(custom_postfix)
    set_logger(logfilename)
    begin_time = arrow.get()
    building_list = args[0]
    source_sample_num_list = args[1]
    prev_data = {'iter_num':0,
                 'next_learning_srcids': get_random_srcids(
                                        building_list,
                                        source_sample_num_list),
                 'model_uuid': None}
    step_datas = iteration_wrapper(iter_num, char2ir_onestep, prev_data, *args)

    building_list = args[0]
    target_building = args[2]
    postfix = 'char2ir_iter' 
    if custom_postfix:
        postfix += '_' + custom_postfix
    with open('result/crf_iter_{0}_{1}.json'\
            .format(''.join(building_list+[target_building]), postfix), 'w') as fp:
        json.dump(step_datas, fp, indent=2)
    end_time = arrow.get()
    print(iter_num, " iterations took: ", end_time - begin_time)


def brick_crfsharp_pretrain():
    # Add Brick tags to the trainer. 
    brick_sentence_dict = dict()
    brick_label_dict = dict()
    with open('metadata/brick_tags_labels.json', 'r') as fp:
        tag_label_list = json.load(fp)
    for tag_labels in tag_label_list:
        char_tags = list(map(itemgetter(0), tag_labels))
        char_labels = list(map(itemgetter(1), tag_labels))
        brick_sentence_dict[''.join(char_tags)] = char_tags
        brick_label_dict[''.join(char_tags)] = char_labels
    brick_srcids = list(brick_sentence_dict.keys())

    brick_label_set = set(reduce(adder, brick_label_dict.values()))
    brick_tags = list(set([brick_label[2:] for brick_label 
                             in brick_label_set]))

    # Collect all tags
    buildings = ['ebu3b', 'bml', 'ghc', 'ap_m']
    building_sentences = []
    building_labels = []
    labels = []
    for building in buildings:
        with open('metadata/{0}_label_dict_justseparate.json'.format(building),
                'r') as fp:
            label_dict = json.load(fp)
        labels += reduce(adder, label_dict.values())
    labels = list(set(labels))
    labels = [label for label in labels if label not in brick_tags]
    for label in labels:
        sentence = [c if c != '_' else 'SPECIAL' for c in label]
        char_labels = ['B_' + label] + ['I_' + label] * (len(label) - 1)
        try:
            assert len(sentence) == len(char_labels)
        except:
            pdb.set_trace()
        building_sentences.append(sentence)
        building_labels.append(char_labels)
    
    # Add samples to the trainer
    trainer = CRFSharp(base_dir = './temp', \
                       template = './model/scrabble.template',
                       thread = thread_num,
                       maxiter = crfsharp_maxiter,
                       nbest = 1)
    sentences = [brick_sentence_dict[srcid] for srcid in brick_srcids] + \
                    building_sentences
    labels = [brick_label_dict[srcid] for srcid in brick_srcids] + \
                 building_labels
    crf_model_file = trainer.encode(sentences, labels, None)
    os.rename(crf_model_file, 'model/brick_crfsharp.model')
    os.rename(crf_model_file + '.feature', 
                 'model/brick_crfsharp.model.feature')
    os.rename(crf_model_file + '.feature.raw_text', 
                 'model/brick_crfsharp.model.feature.raw_text')
    os.rename(crf_model_file + '.alpha', 'model/brick_crfsharp.model.alpha')



if __name__ == '__main__':
    brick_crfsharp_pretrain()
