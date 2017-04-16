import os
from functools import reduce
import json                                                                     
import random
from collections import OrderedDict
import pdb
from copy import deepcopy
from operator import itemgetter
from itertools import islice
import argparse
import logging
from imp import reload

import pycrfsuite                                                               
import pandas as pd                                                             
import numpy as np                    
from IPython.display import Audio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from resulter import Resulter

def save_fig(fig, name, dpi=400):
    pp = PdfPages(name)
    pp.savefig(fig, bbox_inches='tight', pad_inches=0, dpi=dpi)
    pp.close()

def calc_features(sentence, ts_features=None):
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
        if ts_features:
            for j, feat in enumerate(ts_features):
                features['ts_feat_'+str(j)] = feat
        sentenceFeatures.append(features)
    return sentenceFeatures


def learn_crf_model(building_name, 
                    N, 
                    token_type='justseparate', 
                    label_type='label', 
                    use_cluster_flag=False,
                    debug_flag=False,
                    use_brick_flag=False):
    spec = {
            'source_building': building_name,
            'target_building': building_name,
            'source_sample_num': N,
            'label_type': label_type,
            'token_type': token_type,
            'use_cluster_flag': use_cluster_flag
            }
    # It does training and testing on the same building but different points.

    log_filename = 'logs/training_{0}_{1}_{2}_{3}_{4}.log'\
            .format(building_name, N, token_type, label_type, \
                    'clustered' if use_cluster_flag else 'unclustered')
    logging.basicConfig(filename=log_filename, 
                        level=logging.DEBUG,
                        format='%(asctime)s %(message)s')
    logging.info("Started!!!")

    ### TRAINING ###

    data_available_buildings = []

    with open("metadata/%s_char_sentence_dict_%s.json" % (building_name, token_type), "r") as fp:
        sentence_dict = json.load(fp)
    sentence_dict = dict((srcid, [char for char in sentence]) for (srcid, sentence) in sentence_dict.items())
        
    with open('metadata/{0}_char_category_dict.json'.format(building_name), 'r') as fp:
        char_category_dict = json.load(fp)
    with open('metadata/{0}_char_label_dict.json'.format(building_name), 'r') as fp:
        char_label_dict = json.load(fp)
        
    if label_type=='label':
        label_dict = char_label_dict
    elif label_Type=='category':
        label_dict = char_category_dict

    if building_name in data_available_buildings:
        with open("model/fe_%s.json"%building_name, "r") as fp:
            data_feature_dict = json.load(fp)
        with open("model/fe_%s.json"%building_name, "r") as fp:
            normalized_data_feature_dict = json.load(fp)
        for srcid in sentence_dict.keys():
            if not normalized_data_feature_dict.get(srcid):
                normalized_data_feature_dict[srcid] = None

    cluster_filename = 'model/%s_word_clustering_%s.json' % (building_name, token_type)
    if os.path.isfile(cluster_filename):
        with open(cluster_filename, 'r') as fp:
            cluster_dict = json.load(fp)

    # Learning Sample Selection
    SAMPLE_NUM = N

    sample_srcid_list = set()
    length_counter = lambda x:len(x[1])
    ander = lambda x,y:x and y
    labeled_srcid_list = list(label_dict.keys())
    if use_cluster_flag:
        sample_cnt = 0
        sorted_cluster_dict = OrderedDict(
                sorted(cluster_dict.items(), key=length_counter, reverse=True))
        while len(sample_srcid_list) < SAMPLE_NUM:
            for cluster_num, srcid_list in sorted_cluster_dict.items():
                valid_srcid_list = set(srcid_list)\
                        .intersection(set(labeled_srcid_list))\
                        .difference(set(sample_srcid_list))
                if len(valid_srcid_list) > 0:
                    sample_srcid_list.add(\
                            random.choice(list(valid_srcid_list)))
                if len(sample_srcid_list) >= SAMPLE_NUM:
                    break
    else:
        random_idx_list = random.sample(\
                            range(0,len(labeled_srcid_list)),SAMPLE_NUM)
        sample_srcid_list = [labeled_srcid_list[i] for i in random_idx_list]

    # Cluster counting (remove later)
    cluster_counter_dict = dict((cluster_id,0)
                                  for cluster_id 
                                  in cluster_dict.keys())

    for srcid in sample_srcid_list:
        for cluster_id, srcid_list in cluster_dict.items():
            if srcid in srcid_list:
                cluster_counter_dict[cluster_id] += 1


    trainer = pycrfsuite.Trainer(verbose=False, algorithm='pa')
    # algorithm: {‘lbfgs’, ‘l2sgd’, ‘ap’, ‘pa’, ‘arow’}
    trainer.set_params({'feature.possible_states': True,
                        'feature.possible_transitions': True})
    for srcid in sample_srcid_list:
        sentence = list(map(itemgetter(0), label_dict[srcid]))
        labels = list(map(itemgetter(1), label_dict[srcid]))
        if building_name in data_available_buildings:
            data_features = normalized_data_feature_dict[srcid]
        else:
            data_features = None
        trainer.append(pycrfsuite.ItemSequence(
            calc_features(sentence, data_features)), labels)

    # Learn Brick tags

#    if use_brick_flag:
#        with open('metadata/brick_tags_labels.json', 'r') as fp:
#            tag_label_list = json.load(fp)
#        for tag_labels in tag_label_list:
#            char_tags = [' '] + list(map(itemgetter(0), tag_labels)) + [' ']
#            char_labels = ['O'] + list(map(itemgetter(1), tag_labels)) + ['O']
#            trainer.append(pycrfsuite.ItemSequence(
#                calc_features(char_tags, None)), char_labels)

    crf_model_file = 'model/crf_params_char_{0}_{1}_{2}_{3}_{4}.crfsuite'\
            .format(building_name, 
                    token_type, 
                    label_type, 
                    str(SAMPLE_NUM), 
                    'clustered' if use_cluster_flag else 'notclustered')

    # Train and store the model file
    trainer.train(crf_model_file)


    ### TEST ###
    tagger = pycrfsuite.Tagger()
    tagger.open(crf_model_file)

    predicted_dict = dict()
    score_dict = dict()
    for srcid, sentence in sentence_dict.items():
        if building_name in data_available_buildings:
            data_features = normalized_data_feature_dict[srcid]
        else:
            data_features = None
        predicted = tagger.tag(calc_features(sentence, data_features))
        predicted_dict[srcid] = predicted
        score_dict[srcid] = tagger.probability(predicted)

    precisionOfTrainingDataset = 0.0                                                
    totalWordCount = 0.0                                                            
    labeledSrcidList = list(label_dict.keys())
    resulter = Resulter(spec=spec)
    result_dict = dict()

    for srcid in sentence_dict.keys():
        sentence = sentence_dict[srcid]                                              
        predicted = predicted_dict[srcid]
        if not srcid in list(label_dict.keys()):                                       
            for word, predTag in zip(sentence, predicted): 
                if debug_flag:
                    pass
                    #print('{:20s} {:20s}'.format(word,predTag))
                else:
                    pass
        else:          
            if srcid in sample_srcid_list:
                logging.info("=============== %s TRAINING =============== %e" 
                      % (srcid, score_dict[srcid]))
            else:
                logging.info("=============== %s TEST     =============== %e" 
                      % (srcid, score_dict[srcid]))
            printing_pairs = list()
            sentence_label = label_dict[srcid]
            label_list = list(map(itemgetter(1), sentence_label))
            #sentence = list(map(itemgetter(0), sentence_label))
            resulter.add_one_result(srcid, sentence, predicted, label_list)
            
            for word, predTag, origLabel in \
                    zip(sentence, predicted, label_list):
                printing_pair = [word,predTag,origLabel]
                if predTag==origLabel:                                          
                    precisionOfTrainingDataset += 1                             
                    printing_pair = ['O'] + printing_pair                       
                else:                                                           
                    printing_pair = ['X'] + printing_pair                                       
                totalWordCount += 1                                             
                printing_pairs.append(printing_pair)                                
            if 'X' in [pair[0] for pair in printing_pairs]:                         
                for (flag, word, predTag, origLabel) in printing_pairs:
                    if debug_flag:
                        logging.info('{:5s} {:20s} {:20s} {:20s}'\
                                .format(flag, word, predTag, origLabel))                                          
    logging.info('# Learning Sample: {0}'.format(len(sample_srcid_list)))
    logging.info('# Test Sample: {0}'\
                    .format(len(label_dict) - len(sample_srcid_list)))
    logging.info('Precision: {0}'
                    .format(precisionOfTrainingDataset/totalWordCount))
    
    
    result_file = 'result/result_{0}_{1}_{2}_{3}_{4}.json'\
                    .format(building_name, 
                            token_type, 
                            label_type, 
                            str(SAMPLE_NUM),
                            'clustered' if use_cluster_flag else 'unclustered')
    summary_file = 'result/summary_{0}_{1}_{2}_{3}_{4}.json'\
                    .format(building_name, 
                            token_type, 
                            label_type, 
                            str(SAMPLE_NUM),
                            'clustered' if use_cluster_flag else 'unclustered')

    resulter.serialize_result(result_file)
    resulter.summarize_result()
    resulter.serialize_summary(summary_file)
    resulter.store_result_db()

    len(label_list)

    logging.info("Finished!!!")
    sound_file = 'etc/fins_success.wav'
    Audio(url=sound_file, autoplay=True)


def crf_test(source_building_name, 
             target_building_name,
             source_sample_num,
             token_type='justseparate',
             label_type='label',
             use_cluster_flag=False):
    
    spec = {
            'source_building': source_building_name,
            'target_building': target_building_name,
            'source_sample_num': source_sample_num,
            'label_type': label_type,
            'token_type': token_type,
            'use_cluster_flag': use_cluster_flag
            }
    
    resulter = Resulter(spec=spec)
    log_filename = 'logs/test_{0}_{1}_{2}_{3}_{4}_{5}.log'\
            .format(source_building_name, 
                    target_building_name,
                    source_sample_num, 
                    token_type, 
                    label_type, \
                    'clustered' if use_cluster_flag else 'unclustered')
    logging.basicConfig(filename=log_filename, 
                        level=logging.DEBUG,
                        format='%(asctime)s %(message)s')
    logging.info("Started!!!")
    #token_types = ['alphaandnum', 'justseparate']
    label_type = 'label'
    #label_types = ['label', 'category']

    data_available_buildings = []

    with open('metadata/{0}_char_label_dict.json'\
                .format(target_building_name), 'r') as fp:
        target_label_dict = json.load(fp)
    with open('metadata/{0}_char_sentence_dict_{1}.json'\
                .format(target_building_name, token_type), 'r') as fp:
        char_sentence_dict = json.load(fp)
    with open('metadata/{0}_sentence_dict_{1}.json'\
                .format(target_building_name, token_type), 'r') as fp:
        word_sentence_dict = json.load(fp)

    sentence_dict = char_sentence_dict
    sentence_dict = dict((srcid, sentence) 
                         for srcid, sentence 
                         in sentence_dict.items() 
                         if target_label_dict.get(srcid))
    crf_model_file = 'model/crf_params_char_{0}_{1}_{2}_{3}_{4}.crfsuite'\
                        .format(source_building_name, 
                                token_type, 
                                label_type, 
                                str(source_sample_num),
                                'clustered' if use_cluster_flag else 'notclustered')

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
        for word, predTag, origLabel in zip(sentence, predicted, orig_label_list):
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
        error_rate_dict[srcid] = sum([pair[0]=='X' for pair in printing_pairs])/float(len(sentence))
        if 'X' in [pair[0] for pair in printing_pairs]:
            for (flag, word, predTag, origLabel) in printing_pairs:
                logging.info('{:5s} {:20s} {:20s} {:20s}'\
                                .format(flag, word, predTag, origLabel))
    
    result_file = 'result/test_result_{0}_{1}_{2}_{3}_{4}_{5}.json'\
                    .format(source_building_name,
                            target_building_name,
                            token_type, 
                            label_type, 
                            source_sample_num,
                            'clustered' if use_cluster_flag else 'unclustered')
    summary_file = 'result/test_summary_{0}_{1}_{2}_{3}_{4}_{5}.json'\
                    .format(source_building_name, 
                            target_building_name,
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
                            target_building_name,
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

def str2bool(v):
    if v in ['true', 'True']:
        return True
    elif v in ['false', 'False']:
        return False
    else:
        assert(False)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)

    parser.add_argument(choices=['learn', 'predict'],
                        dest = 'prog')
    
    parser.add_argument('predict', 
                         action='store_true',
                         default=False)

    parser.add_argument('-b', 
                        type=str, 
                        help='Learning source building name',
                        dest='source_building')
    parser.add_argument('-n', 
                        type=int, 
                        help='The number of learning sample',
                        dest='sample_num')
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
    parser.add_argument('-ub', 
                        type='bool', 
                        help='Use Brick when learning',
                        default=False,
                        dest='use_brick_flag')
    
    args = parser.parse_args()

    if args.prog=='learn':
        learn_crf_model(building_name = args.source_building, 
                        N = args.sample_num, 
                        token_type = 'justseparate',
                        label_type = args.label_type,
                        use_cluster_flag = args.use_cluster_flag,
                        debug_flag = args.debug_flag,
                        use_brick_flag=args.use_brick_flag)
    elif args.prog=='predict':
        crf_test(source_building_name = args.source_building, 
                 target_building_name = args.target_building,
                 source_sample_num = args.sample_num,
                 token_type='justseparate',
                 label_type = args.label_type,
                 use_cluster_flag = args.use_cluster_flag)
    else:
        print('Either learn or predict should be provided')
        assert(False)
