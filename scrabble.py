import os
from functools import reduce
import json
import random
from collections import OrderedDict, defaultdict
import pdb
from copy import deepcopy
from operator import itemgetter
from itertools import islice
import argparse
import logging
from imp import reload
from uuid import uuid4 as gen_uuid

import pycrfsuite
import pandas as pd
import numpy as np
from IPython.display import Audio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from bson.binary import Binary as BsonBinary
import arrow

from resulter import Resulter
from mongo_models import store_model, get_model, get_tags_mapping
from entity_recognition import learn_brick_tagsets, \
                               test_brick_tagset, \
                               batch_test_brick_tagset

temp_dir = 'temp'
if not os.path.isdir(temp_dir):
    os.mkdir(temp_dir)


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
            .format(building_list[0], source_sample_num_list[0], token_type, label_type, \
                    'clustered' if use_cluster_flag else 'unclustered')
    logging.basicConfig(filename=log_filename, 
                        level=logging.DEBUG,
                        format='%(asctime)s %(message)s')
    logging.info("Started!!!")

    ### TRAINING ###

    trainer = pycrfsuite.Trainer(verbose=False, algorithm='pa')
    # algorithm: {‘lbfgs’, ‘l2sgd’, ‘ap’, ‘pa’, ‘arow’}
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

        if label_type=='label':
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

        # Cluster counting (remove later)
        cluster_counter_dict = dict((cluster_id,0)
                                      for cluster_id 
                                      in cluster_dict.keys())

        for srcid in sample_srcid_list:
            for cluster_id, srcid_list in cluster_dict.items():
                if srcid in srcid_list:
                    cluster_counter_dict[cluster_id] += 1


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
            'gen_time': arrow.get().datetime,
            'use_cluster_flag': use_cluster_flag,
            'token_type': 'justseparate',
            'label_type': 'label',
            'model_binary': BsonBinary(model_bin),
            'source_building_count': len(building_list)
            }
    store_model(model)
    os.remove(crf_model_file)

    logging.info("Finished!!!")
    sound_file = 'etc/fins_success.wav'
    Audio(url=sound_file, autoplay=True)


def crf_test(building_list, 
        source_sample_num_list, 
        target_building,
        token_type='justseparate', 
        label_type='label', 
        use_cluster_flag=False,
        use_brick_flag=False):
    assert(len(building_list)==len(source_sample_num_list))


    source_building_name = building_list[0] #TODO: remove this to use the list

    model_query = {'$and':[]}
    model_metadata = {
            'label_type': label_type,
            'token_type': token_type,
            'use_cluster_flag': use_cluster_flag,
            'source_building_count': len(building_list)
            }
    result_metadata = deepcopy(model_metadata)
    result_metadata['source_cnt_list'] = []
    result_metadata['target_building'] = target_building
    for building, source_sample_num in \
            zip(building_list, source_sample_num_list):
        model_query['$and'].append(
                {'source_list.{0}'.format(building): {'$exists': True}})
        model_query['$and'].append({'$where': 
            'this.source_list.{0}.length={1}'.\
                    format(building, source_sample_num)})
        result_metadata['source_cnt_list'].append([building, source_sample_num])
    model_query['$and'].append(model_metadata)
    model_query['$and'].append({'source_building_count':len(building_list)})
    model = get_model(model_query)
    
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
    #return dict((k,v) for k, v in d.items() if k in ks)
    return dict([(k,d[k]) for k in ks])

def entity_recognition_from_ground_truth(building_list,
        source_sample_num_list,
        target_building,
        token_type='justseparate',
        label_type='label',
        use_cluster_flag=False,
        use_brick_flag=False):
    assert(len(building_list)==len(source_sample_num_list))

    ## Learning stage
    # construct source data information data structure
    source_cnt_list = [[building,cnt]\
                       for building, cnt\
                       in zip(building_list, source_sample_num_list)]


    # Construct inputs for learning a classifier
    learning_sentence_dict = dict()
    token_label_dict = dict()
    learning_truths_dict = dict()
    sample_srcid_list_dict = dict()
    for building, sample_num in zip(building_list, source_sample_num_list):
        with open('metadata/{0}_char_sentence_dict_{1}.json'\
                  .format(building, token_type), 'r') as fp:
            sentence_dict = json.load(fp)
        with open('metadata/{0}_char_label_dict.json'\
                  .format(building), 'r') as fp:
            sentence_label_dict = json.load(fp)
        with open('metadata/{0}_ground_truth.json'\
                  .format(building), 'r') as fp:
            truths_dict = json.load(fp)
        sample_srcid_list = random.sample(sentence_label_dict.keys(), sample_num)
        sample_srcid_list_dict[building] = sample_srcid_list
        learning_sentence_dict.update(\
            sub_dict_by_key_set(sentence_dict, sample_srcid_list))
        label_dict = dict((srcid, list(map(itemgetter(1), labels))) \
                          for srcid, labels in sentence_label_dict.items())
        token_label_dict.update(\
            sub_dict_by_key_set(label_dict, sample_srcid_list))
        learning_truths_dict.update(\
            sub_dict_by_key_set(truths_dict, sample_srcid_list))

    # Correct so far.
    tagset_classifier, tagset_vectorizer = learn_brick_tagsets(\
                                                learning_sentence_dict,\
                                                token_label_dict,\
                                                learning_truths_dict)

    ## Test stage
    # get test dataset
    with open('metadata/{0}_char_sentence_dict_{1}.json'\
              .format(target_building, token_type), 'r') as fp:
        sentence_dict = json.load(fp)
    with open('metadata/{0}_char_label_dict.json'\
              .format(target_building), 'r') as fp:
        sentence_label_dict = json.load(fp)
    with open('metadata/{0}_ground_truth.json'\
              .format(target_building), 'r') as fp:
        truths_dict = json.load(fp)
    test_srcid_list = [srcid for srcid in sentence_label_dict.keys() \
                       if srcid not in sample_srcid_list_dict[target_building]]
    test_sentence_dict = sub_dict_by_key_set(sentence_dict, test_srcid_list)
    token_label_dict = dict((srcid, list(map(itemgetter(1), labels))) \
                            for srcid, labels in sentence_label_dict.items())
    test_token_label_dict = sub_dict_by_key_set(token_label_dict, test_srcid_list)
#    truths_dict = sub_dict_by_key_set(truths_dict, test_srcid_list)


    correct_cnt = 0
    """
    for srcid in test_srcid_list:
        #pdb.set_trace()
        pred_dict[srcid] = test_brick_tagset(sentence=test_sentence_dict[srcid],\
                                         token_labels=test_token_label_dict[srcid],\
                                         classifier=tagset_classifier,\
                                         vectorizer=tagset_vectorizer)
        if set(truths_dict[srcid]) == set(pred_dict[srcid]):
            correct_cnt += 1
    """
    pred_tagsets_dict, prob_dict = batch_test_brick_tagset(test_sentence_dict,\
                                                test_token_label_dict,\
                                                tagset_classifier,
                                                tagset_vectorizer)
    result_dict = defaultdict(dict)
    incorrect_tagsets_dict = dict()
    for srcid, pred_tagsets in pred_tagsets_dict.items():
        one_result = {
            'tagsets': pred_tagsets,
            'certainty': prob_dict[srcid]
        }
        if set(truths_dict[srcid]) == set(pred_tagsets):
            correct_cnt += 1
            result_dict['correct'][srcid] = one_result
            #result_dict['correct'][srcid] = pred_tagsets
        else:
            result_dict['incorrect'][srcid] = one_result

    print('precision')
    print(float(correct_cnt) / len(test_srcid_list))
    with open('result/tagset_{0}.json'.format(building), 'w') as fp:
        json.dump(result_dict, fp, indent=2)
#    with open('result/incorrect_tagset_{0}.json'.format(building), 'w') as fp:
#        json.dump(incorrect_tagsets_dict, fp, indent=2)


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

    parser.add_argument(choices=['learn', 'predict', 'entity'],
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
    parser.add_argument('-ub', 
                        type='bool', 
                        help='Use Brick when learning',
                        default=False,
                        dest='use_brick_flag')
    
    args = parser.parse_args()

    if args.prog=='learn':
        learn_crf_model(building_list = args.source_building_list, 
                        source_sample_num_list = args.sample_num_list, 
                        token_type = 'justseparate',
                        label_type = args.label_type,
                        use_cluster_flag = args.use_cluster_flag,
                        debug_flag = args.debug_flag,
                        use_brick_flag=args.use_brick_flag)
    elif args.prog=='predict':
        crf_test(building_list = args.source_building_list, 
                source_sample_num_list = args.sample_num_list, 
                target_building = args.target_building,
                token_type = 'justseparate',
                label_type = args.label_type,
                use_cluster_flag = args.use_cluster_flag,
                use_brick_flag=args.use_brick_flag)
    elif args.prog=='entity':
        entity_recognition_from_ground_truth(building_list = args.source_building_list,
                source_sample_num_list = args.sample_num_list, 
                target_building = args.target_building,
                token_type = 'justseparate',
                label_type = args.label_type,
                use_cluster_flag = args.use_cluster_flag,
                use_brick_flag=args.use_brick_flag)
    else:
        print('Either learn or predict should be provided')
        assert(False)
