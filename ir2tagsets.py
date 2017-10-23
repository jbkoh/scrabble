from collections import defaultdict, Counter, OrderedDict
from operator import itemgetter
from copy import deepcopy
from itertools import chain
from uuid import uuid4
def gen_uuid():
    return str(uuid4())
from pprint import PrettyPrinter
pp = PrettyPrinter()
from multiprocessing import Pool, Manager, Process
import pdb
import math
import pickle

import arrow
import numpy as np
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
from scipy.stats import entropy as get_entropy
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from skmultilearn.ensemble import RakelO, LabelSpacePartitioningClassifier
from skmultilearn.problem_transform import LabelPowerset, ClassifierChain, \
                                           BinaryRelevance
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA

from common import *
from eval_func import *
from brick_parser import pointTagsetList        as  point_tagsets,\
                         locationTagsetList     as  location_tagsets,\
                         equipTagsetList        as  equip_tagsets,\
                         pointSubclassDict      as  point_subclass_dict,\
                         equipSubclassDict      as  equip_subclass_dict,\
                         locationSubclassDict   as  location_subclass_dict,\
                         tagsetTree             as  tagset_tree
from hcc import StructuredClassifierChain
from mongo_models import store_model, get_model, get_tags_mapping, \
                         get_crf_results, store_result, get_entity_results
from char2ir import crf_test, learn_crf_model
from time_series_to_ir import TimeSeriesToIR

# Init Brick schema with customizations.

total_srcid_dict = dict()

tagset_list = point_tagsets + location_tagsets + equip_tagsets
tagset_list.append('networkadapter')

subclass_dict = dict()
subclass_dict.update(point_subclass_dict)
subclass_dict.update(equip_subclass_dict)
subclass_dict.update(location_subclass_dict)
subclass_dict['networkadapter'] = list()
subclass_dict['unknown'] = list()
subclass_dict['none'] = list()

ts_feature_filename = 'TS_Features/features.pkl'

#tagset_classifier_type = 'RandomForest'
tagset_classifier_type = 'StructuredCC'
#tagset_classifier_type = 'CC'

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

def extend_tree(tree, k, d):
    for curr_head, branches in tree.items():
        if k==curr_head:
            branches.append(d)
        for branch in branches:
            extend_tree(branch, k, d)

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

def augment_ts(phrase_dict, srcids, ts2ir):
    with open(ts_feature_filename, 'rb') as fp:
        ts_features = pickle.load(fp, encoding='bytes')
    ts_tags_pred = ts2ir.predict(ts_features, srcids)

    tag_binarizer = ts2ir.get_binarizer()
    pred_tags_list = tag_binarizer.inverse_transform(ts_tags_pred)

    for srcid, pred_tags in zip(srcids, pred_tags_list):
        phrase_dict[srcid] += list(pred_tags)
    return phrase_dict

def extend_tagset_list(new_tagsets):
    global tagset_list
    tagset_list.extend(new_tagsets)
    tagset_list = list(set(tagset_list))

def tree_flatter(tree, init_flag=True):
    branches_list = list(tree.values())
    d_list = list(tree.keys())
    for branches in branches_list:
        for branch in branches:
            added_d_list = tree_flatter(branch)
            d_list = [d for d in d_list if d not in added_d_list]\
                    + added_d_list
    return d_list

def entity_recognition_from_ground_truth(building_list,
                                         source_sample_num_list,
                                         target_building,
                                         use_cluster_flag=False,
                                         use_brick_flag=False,
                                         debug_flag=False,
                                         eda_flag=False,
                                         ts_flag=False,
                                         negative_flag=True,
                                         n_jobs=4,
                                         inc_num=10,
                                         query_strategy='phrase_util',
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
                                        ):
    """
    Input:
        building_list: learning building lists
        source_sample_num_list: # samples from each building in building_list
        target_building: target building's name
        use_cluster_flag: whether to use bow clustering to select samples
        use_brick_flag: whether to use brick tagsets when learning or not
        debug_flag: debugging mode or not. May call pdb and/or print more.
        eda_flag: whether to use EasyDomainAdaptation algorithm or not.
        ts_flag: whether to exploit timeseries features or not
        negative_flag: whether to generate negative examples or not.
        n_jobs: parallel processing CPU num.
        prev_step_data: for iteration. default = {
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

    """


    logging.info('Entity Recognition Get Started.')
    global tagset_list
    global total_srcid_dict
    global tree_depth_dict
    assert len(building_list) == len(source_sample_num_list)

    ########################## DATA INITIATION ##########################
    # construct source data information data structure
    source_cnt_list = [[building, cnt]\
                       for building, cnt\
                       in zip(building_list, source_sample_num_list)]
    if not prev_step_data['metadata']:
        metadata = {
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
    print('################ Iteration {0} ################'.format(iter_cnt))

    ### Get Learning Data
    sample_srcid_list_dict = dict()
    validation_srcids = []
    validation_truths_dict = {}
    for building, sample_num in zip(building_list, source_sample_num_list):
        with open('metadata/{0}_char_label_dict.json'\
                  .format(building), 'r') as fp:
            sentence_label_dict = json.load(fp) # char-level labels
        srcids = list(sentence_label_dict.keys())
        # If it is the first iteration, select random samples.
        if iter_cnt == 1:
            sample_srcid_list = select_random_samples(\
                                    building,\
                                    sentence_label_dict.keys(),\
                                    sample_num, \
                                    use_cluster_flag,\
                                    reverse=True,
                                    shuffle_flag=False)
            sample_srcid_list_dict[building] = sample_srcid_list
            learning_srcids += sample_srcid_list
            total_srcid_dict[building] = list(sentence_label_dict.keys())
        # Otherwise, choose based on given learning_srcids
        else:
            sample_srcid_list_dict[building] = [srcid for srcid\
                                                in srcids \
                                                if srcid in learning_srcids]
        # Get validation samples
        # TODO: This is incorrect and maybe fixed later.
        #       No validation process for now.
        validation_num = min(len(sentence_label_dict)
                             - len(sample_srcid_list_dict[building]),
                             len(sample_srcid_list_dict[building]))
        validation_srcids += select_random_samples(\
                                    building,\
                                    srcids,\
                                    validation_num,\
                                    use_cluster_flag,\
                                    reverse=True,
                                    shuffle_flag=False)
    _, _, validation_truths_dict, _ = get_multi_buildings_data(building_list,\
                                                            validation_srcids, \
                                                            eda_flag)
    learning_sentence_dict, \
    learning_token_label_dict, \
    learning_truths_dict, \
    phrase_dict = get_multi_buildings_data(building_list, learning_srcids)
    # found_points is just for debugging
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
    test_srcid_dict = get_building_data(target_building, test_srcids)

    # Include tagsets not defined in Brick but added by the ground truth.
    # This better be removed if Brick is complete.
    extend_tagset_list(reduce(adder, \
                [learning_truths_dict[srcid] for srcid in learning_srcids]\
                + [test_truths_dict[srcid] for srcid in test_srcids], []))
    # Augment the tagset tree based on the newly added tagsets
    augment_tagset_tree(tagset_list)
    tree_depth_dict = calc_leaves_depth(tagset_tree)
    assert tree_depth_dict['supply_air_static_pressure_integral_time_setpoint']\
            > 3

    ###########################  LEARNING  ################################
    ### Learning IR-Tagsets model
    source_target_buildings = list(set(building_list + [target_building]))
    begin_time = arrow.get()
    tagset_classifier, tagset_vectorizer, tagset_binarizer, \
            point_classifier, ts2ir = \
            build_tagset_classifier(building_list, target_building,\
                            test_sentence_dict,\
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

    ####################      TEST      #################
    ### Test on Brick
    brick_doc = []
    if use_brick_flag:
        brick_phrase_dict = dict([(str(i), tagset.split('_')) for i, tagset\
                                  in enumerate(tagset_list)])
        brick_srcids = list(brick_phrase_dict.keys())
        brick_pred_tagsets_dict, \
        brick_pred_certainty_dict, \
        brick_pred_point_dict, \
        _ =                 tagsets_prediction(
                                   tagset_classifier, 
                                   tagset_vectorizer,
                                   tagset_binarizer,
                                   brick_phrase_dict,
                                   list(brick_phrase_dict.keys()),
                                   source_target_buildings,
                                   eda_flag,
                                   point_classifier
                                  )

    ### Test on the learning samples
    learning_pred_tagsets_dict, \
    learning_pred_certainty_dict, \
    learning_pred_point_dict, \
    _                     = tagsets_prediction(\
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
                                              learning_pred_point_dict,
                                              phrase_dict,
                                              debug_flag,
                                              tagset_classifier,
                                              tagset_vectorizer)

    ### Test on the entire target building
    target_srcids = raw_srcids_dict[target_building]
    _,\
    _,\
    target_phrase_dict,\
    target_truths_dict,\
    _                   = get_building_data(target_building, target_srcids)
    target_pred_tagsets_dict, \
    target_pred_certainty_dict, \
    target_pred_point_dict, \
    target_prob_mat = tagsets_prediction(\
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
    logging.info('# of not detected tagsets: {0}'.format(list(map(len,
                    next_step_data['unfound_tagsets_history']))))
    logging.info('# of learnt but not detected tagsets: {0}'.format(list(map(len,
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
    print('history of macro f1: {0}'\
          .format(next_step_data['macro_f1_history']))

    ###### Post processing to select next step learning srcids
    """
    if query_strategy == 'util':
        phrase_usages = list(target_result_dict['phrase_usage'].values())
        mean_usage_rate = np.mean(phrase_usages)
        std_usage_rate = np.std(phrase_usages)
        # Select underexploited sentences.
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
                              reverse=True,
                              cluster_dict=cluster_srcid_dict,
                              shuffle_flag=False
                             )
    """
    if query_strategy == 'phrase_util':
        todo_srcids = ir2tagset_al_query_samples_phrase_util(
                            test_srcids,
                            target_building,
                            test_sentence_dict,
                            test_phrase_dict,
                            target_pred_tagsets_dict,
                            inc_num,
                            learning_srcids)
    elif query_strategy == 'strict_phrase_util':
        todo_srcids = ir2tagset_al_query_samples_strict_phrase_util(
                            test_srcids,
                            target_building,
                            test_sentence_dict,
                            test_phrase_dict,
                            target_pred_tagsets_dict,
                            inc_num,
                            learning_srcids)
    elif query_strategy == 'entropy':
        entropies = get_entropy(target_prob_mat.T)
        sorted_entropies = sorted([(target_srcids[i], ent) for i, ent 
                                   in enumerate(entropies)], key=itemgetter(1))
        cluster_dict = get_cluster_dict(target_building)
        added_cids = []
        todo_srcids = []
        new_srcid_cnt = 0
        for srcid, _ in sorted_entropies:
            if srcid in learning_srcids:
                continue
            the_cid = None
            for cid, cluster in cluster_dict.items():
                if srcid in cluster:
                    the_cid = cid
                    break
            if the_cid in added_cids:
                continue
            added_cids.append(the_cid)
            todo_srcids.append(srcid)
            new_srcid_cnt += 1
            if new_srcid_cnt == inc_num:
                break
    elif query_strategy == 'cmn':
        conf_mat = np.abs(target_prob_mat - 0.5)
        min_values = np.min(conf_mat, 1)
        max_values = np.max(conf_mat, 1)
        sorted_min_values = sorted([(target_srcids[i], min_val) for i, min_val 
                                in enumerate(min_values)], key=itemgetter(1))
        sorted_max_values = sorted([(target_srcids[i], max_val) for i, max_val 
                                in enumerate(max_values)], key=itemgetter(1))
        cluster_dict = get_cluster_dict(target_building)
        added_cids = []
        todo_srcids = []
        """
        new_srcid_cnt = 0
        for srcid, _ in sorted_max_values:
            if srcid in learning_srcids:
                continue
            the_cid = None
            for cid, cluster in cluster_dict.items():
                if srcid in cluster:
                    the_cid = cid
                    break
            if the_cid in added_cids:
                continue
            added_cids.append(the_cid)
            todo_srcids.append(srcid)
            new_srcid_cnt += 1
            if new_srcid_cnt == math.floor(inc_num):
            #if new_srcid_cnt == math.floor(inc_num / 2):
                break
            """
        new_srcid_cnt = 0
        for srcid, _ in sorted_min_values:
            if srcid in learning_srcids:
                continue
            the_cid = None
            for cid, cluster in cluster_dict.items():
                if srcid in cluster:
                    the_cid = cid
                    break
            if the_cid in added_cids:
                continue
            added_cids.append(the_cid)
            todo_srcids.append(srcid)
            new_srcid_cnt += 1
            if new_srcid_cnt == math.ceil(inc_num / 2):
                break
    else:
        assert False

    tot_todo_tagsets =  Counter(reduce(adder, [test_truths_dict[srcid] \
                                               for srcid in todo_srcids], []))
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

    # TODO: Validate this
    next_step_data['learning_srcids'] = learning_srcids + todo_srcids * 4

    try:
        del next_step_data['_id']
    except:
        pass
    with open('result/entity_iteration.json', 'w') as fp:
        json.dump(next_step_data, fp, indent=2)
    return target_result_dict, next_step_data


#TODO: Make this more generic to apply to other functions
def entity_recognition_from_ground_truth_get_avg(N,
                                                 building_list,
                                                 source_sample_num_list,
                                                 target_building,
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

def entity_recognition_iteration(iter_num, postfix, *args):
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
    building_list = args[0]
    target_building = args[2]
    step_datas = []
    step_datas.append(step_data)
    for i in range(0, iter_num):
        _, step_data = entity_recognition_from_ground_truth(
                              building_list = building_list,
                              source_sample_num_list = args[1],
                              target_building = target_building,
                              use_cluster_flag = args[3],
                              use_brick_flag = args[4],
                              debug_flag = args[5],
                              eda_flag = args[6],
                              ts_flag = args[7],
                              negative_flag = args[8],
                              n_jobs = args[9],
                              inc_num = args[10],
                              query_strategy = args[11],
                              prev_step_data = step_data
                            )
        step_datas.append(deepcopy(step_data))
    store_result(step_data)
    with open('result/entity_iter_{0}_{1}.json'\
            .format(''.join(building_list+[target_building]), postfix), 'w') as fp:
        json.dump(step_datas, fp, indent=2)

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
        temp_srcid_dict = get_building_data(building, srcids)
        sentence_dict.update(temp_sentence_dict)
        token_label_dict.update(temp_token_label_dict)
        truths_dict.update(temp_truths_dict)
        phrase_dict.update(temp_phrase_dict)

    assert set(srcids) == set(phrase_dict.keys())
    return sentence_dict, token_label_dict, truths_dict, phrase_dict


def get_building_data(building, srcids):
    with open('metadata/{0}_char_sentence_dict.json'\
              .format(building), 'r') as fp:
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
    truths_dict = sub_dict_by_key_set(truths_dict, srcids)
    phrase_dict = make_phrase_dict(sentence_dict, token_label_dict)

    return sentence_dict, token_label_dict, phrase_dict,\
            truths_dict, srcid_dict

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

    # Update TagSet pool to include TagSets not in Brick.
    orig_sample_num = len(learning_srcids)
    new_tagset_list = tree_flatter(tagset_tree, [])
    new_tagset_list = new_tagset_list + [ts for ts in tagset_list \
                                     if ts not in new_tagset_list]
    tagset_list = new_tagset_list
    tagset_binarizer = MultiLabelBinarizer(tagset_list)
    tagset_binarizer.fit([tagset_list])
    assert tagset_list == tagset_binarizer.classes_.tolist()

    # Point only classifier. Just for testing.
    point_classifier = RandomForestClassifier(n_estimators=10, n_jobs=n_jobs)

    ## Init brick tag_list
    raw_tag_list = list(set(reduce(adder, map(splitter, tagset_list))))
    tag_list = deepcopy(raw_tag_list)


    """
    # Extend tag_list with prefixes
    if eda_flag:
        for building in set(building_list + [target_building]):
            prefixer = build_prefixer(building)
            building_tag_list = list(map(prefixer, raw_tag_list))
            tag_list = tag_list + building_tag_list
    """
    # All possible vocabularies.
    vocab_dict = dict([(tag, i) for i, tag in enumerate(tag_list)])

    # Define Vectorizer
    tokenizer = lambda x: x.split()
    # TODO: We could use word embedding like word2vec here instead.
    tagset_vectorizer = TfidfVectorizer(tokenizer=tokenizer,\
                                        vocabulary=vocab_dict)
    #tagset_vectorizer = MeanEmbeddingVectorizer(tokenizer=tokenizer, 
    #                                            vocabulary=vocab_dict)
    #tagset_vectorizer = CountVectorizer(tokenizer=tokenizer,\
    #                                    vocabulary=vocab_dict)

    # Dataset only for points. Just for testing.
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

    # Augment vocab frequencies with timeseries features.
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


    ## Init Brick samples.
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

    # Apply Easy-Domain-Adaptation mechanism. Not useful.
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
        # Make TagSet vectors.
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

    ## Learn the classifier. StructuredCC is the default model.
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

    elif tagset_classifier_type == 'CC':
        def meta_cc(**kwargs):
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
            tagset_classifier = ClassifierChain(classifier=base_classifier)
            return tagset_classifier
        meta_classifier = meta_cc
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
    elif tagset_classifier_type == 'StructuredCC_RF':
        base_classifier = RandomForest()
        tagset_classifier = StructuredClassifierChain(base_classifier,
                                                      tagset_binarizer,
                                                      subclass_dict,
                                                      tagset_vectorizer.vocabulary,
                                                      n_jobs)
    elif tagset_classifier_type == 'StructuredCC_LinearSVC':
        def meta_scc_svc(**kwargs):
            base_classifier = LinearSVC(loss='hinge', tol=1e-5,\
                                        max_iter=2000, C=2,
                                        fit_intercept=False,
                                        class_weight='balanced')
            tagset_classifier = StructuredClassifierChain(base_classifier,
                                                          tagset_binarizer,
                                                          subclass_dict,
                                                          tagset_vectorizer.vocabulary,
                                                          n_jobs)
            return tagset_classifier
        params_list_dict = {}
        meta_classifier = meta_scc_svc
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

    # TODO: This was for hyper-parameter optimization.
    #       But I disabled it because it's too slow.
    tagset_classifier = parameter_validation(learning_vect_doc[:orig_sample_num],
                         truth_mat[:orig_sample_num],
                         orig_learning_srcids,
                         params_list_dict, meta_classifier, tagset_vectorizer,
                         tagset_binarizer, source_target_buildings, eda_flag)

    # Actual fitting.
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
    prob_mat = classifier.predict_proba(vect_doc)
    logging.info('Finished Tagset prediction')
    if not isinstance(pred_mat, np.ndarray):
        try:
            pred_mat = pred_mat.toarray()
        except:
            pred_mat = np.asarray(pred_mat)
    logging.info('Start point prediction')
    logging.info('Finished point prediction')
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
    return pred_tagsets_dict, pred_certainty_dict, pred_point_dict, prob_mat


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
    #_, _, macro_f1, _ = precision_recall_fscore_support(\
    #                        eval_binarizer.transform(true_tagsets_list),\
    #                        eval_binarizer.transform(pred_tagsets_list),\
    #                        average='macro')
    macro_f1 = get_macro_f1(eval_binarizer.transform(true_tagsets_list),
                            eval_binarizer.transform(pred_tagsets_list))
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

def accuracy_func(pred_tagsets, true_tagsets, labels=None):
    pred_tagsets = set(pred_tagsets)
    true_tagsets = set(true_tagsets)
    return len(pred_tagsets.intersection(true_tagsets))\
            / len(pred_tagsets.union(true_tagsets))

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

def hamming_loss_func(pred_tagsets, true_tagsets, labels):
    incorrect_cnt = 0
    for tagset in pred_tagsets:
        if tagset not in true_tagsets:
            incorrect_cnt += 1
    for tagset in true_tagsets:
        if tagset not in pred_tagsets:
            incorrect_cnt += 1
    return incorrect_cnt / len(labels)

def subset_accuracy_func(pred_Y, true_Y, labels):
    return 1 if set(pred_Y) == set(true_Y) else 0

def get_micro_f1(true_mat, pred_mat):
    TP = np.sum(np.bitwise_and(true_mat==1, pred_mat==1))
    TN = np.sum(np.bitwise_and(true_mat==0, pred_mat==0))
    FN = np.sum(np.bitwise_and(true_mat==1, pred_mat==0))
    FP = np.sum(np.bitwise_and(true_mat==0, pred_mat==1))
    micro_prec = TP / (TP + FP)
    micro_rec = TP / (TP + FN)
    return 2 * micro_prec * micro_rec / (micro_prec + micro_rec)

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

def crf_entity_recognition_iteration(iter_num, postfix, *args):
    building_list = args[0]
    target_building = args[2]
    step_datas = iteration_wrapper(iter_num, entity_recognition_from_crf, *args)
    with open('result/crf_entity_iter_{0}_{1}.json'\
            .format(''.join(building_list+[target_building]), postfix), 'w') as fp:
        json.dump(step_datas, fp, indent=2)

def entity_recognition_from_crf(prev_step_data,\
                                building_list,\
                                source_sample_num_list,\
                                target_building,\
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
    inc_num = 10

    ### Initialize CRF Data
    crf_result_query = {
        'use_cluster_flag': use_cluster_flag,
        'building_list': building_list,
        'source_sample_num_list': source_sample_num_list,
        'target_building': target_building,
    }
    # Read previous step's configuration
    if prev_step_data.get('learning_srcids_history'):
        iter_cnt = len(prev_step_data['learning_srcids_history'])
        crf_result_query['learning_srcids'] = \
                prev_step_data['learning_srcids_history'][-1]
        target_idx = building_list.index(target_building)
        crf_result_query['source_sample_num_list'] = source_sample_num_list

    # Read CRF result
    crf_result = get_crf_results(crf_result_query)
    # If not exists, generate one.
    if not crf_result:
        if crf_result_query.get('learning_srcids'):
            learning_srcids = sorted(crf_result_query['learning_srcids'])
        else:
            learning_srcids = []
        # Learn CRF model if necessary
        learn_crf_model(building_list,
                    source_sample_num_list,
                    use_cluster_flag,
                    use_brick_flag,
                    {
                        'learning_srcids': deepcopy(learning_srcids),
                        'iter_cnt': prev_step_data['iter_num']
                    })
        # Label BILOUS tags based on the above model.
        crf_test(building_list,
                 source_sample_num_list,
                 target_building,
                 use_cluster_flag,
                 use_brick_flag,
                 learning_srcids)
        crf_result = get_crf_results(crf_result_query)
        if not crf_result:
            pdb.set_trace()
            crf_result = get_crf_results(crf_result_query)
        assert crf_result

    # given_srcids == srcids labeled by CRF == entire target set
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
    _,_,_, crf_truths_dict, _ = get_building_data(target_building, crf_srcids)
    given_sentence_dict, \
    given_token_label_dict, \
    given_truths_dict, \
    given_phrase_dict = get_multi_buildings_data(building_list, given_srcids)

    # Add tagsets in the learning dataset if not exists.
    extend_tagset_list(reduce(adder, \
                [given_truths_dict[srcid] for srcid in given_srcids]\
                + [crf_truths_dict[srcid] for srcid in crf_srcids], []))
    augment_tagset_tree(tagset_list)
    source_target_buildings = list(set(building_list + [target_building]))

    # Learning IR->Tagset model
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
    tree_depth_dict = calc_leaves_depth(tagset_tree)

    # Infer Tagsets based on the above model.
    crf_pred_tagsets_dict, \
    crf_pred_certainty_dict, \
    crf_pred_point_dict = tagsets_prediction(\
                                   classifier, vectorizer, \
                                   binarizer, crf_phrase_dict, \
                                   crf_srcids, source_target_buildings,
                                   eda_flag, point_classifier, ts2ir)

    # Calculate utilization
    crf_token_usage_dict = determine_used_tokens_multiple(\
                                crf_sentence_dict, crf_token_label_dict, \
                                crf_pred_tagsets_dict, crf_srcids)
    crf_token_usage_rate_dict = dict((srcid, sum(usage)/len(usage))\
                                     for srcid, usage \
                                     in crf_token_usage_dict.items())
    usage_rates = list(crf_token_usage_rate_dict.values())
    usage_rate_mean = np.mean(usage_rates)
    usage_rate_mean = np.std(usage_rates)

    # Evaluate the result
    crf_entity_result_dict = tagsets_evaluation(crf_truths_dict, crf_pred_tagsets_dict,
                       crf_pred_certainty_dict, crf_srcids,
                       crf_pred_point_dict, crf_phrase_dict, debug_flag)
    # Select srcids to ask for the next step
    todo_srcids = find_todo_srcids(crf_token_usage_rate_dict, crf_srcids, \
                                   inc_num, crf_sentence_dict, target_building)
    curr_learning_srcids = sorted(reduce(adder, crf_result['source_list']\
                                         .values()))
    updated_learning_srcids = sorted(todo_srcids + curr_learning_srcids)
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

    next_step_data['learning_srcids_history'].append(updated_learning_srcids)
    next_step_data['result']['entity'].append(crf_entity_result_dict)
    next_step_data['result']['crf'].append(crf_result)
    return next_step_data

def determine_used_tokens_multiple(sentence_dict, token_label_dict, \
                                   tagsets_dict, srcids):
    token_usage_dict = dict()
    for srcid in srcids:
        token_usage_dict[srcid] = determine_used_tokens(\
                                        sentence_dict[srcid],\
                                        token_label_dict[srcid],\
                                        tagsets_dict[srcid])
    return token_usage_dict

def ir2tagset_al_query_samples_cmn(prob_mat,
                                    target_srcids,
                                    learning_srcids,
                                    target_building):

    conf_mat = np.abs(prob_mat - 0.5)
    min_values = np.min(conf_mat, 1)
    max_values = np.max(conf_mat, 1)
    sorted_min_values = sorted([(target_srcids[i], min_val) for i, min_val 
                                in enumerate(min_values)
                                if target_srcids[i] not in learning_srcids], 
                               key=itemgetter(1))
    cluster_dict = get_cluster_dict(target_building)
    added_cids = []
    todo_srcids = []
    new_srcid_cnt = 0
    for srcid, _ in sorted_max_values:
        if srcid in learning_srcids:
            continue
        the_cid = None
        for cid, cluster in cluster_dict.items():
            if srcid in cluster:
                the_cid = cid
                break
        if the_cid in added_cids:
            continue
        added_cids.append(the_cid)
        todo_srcids.append(srcid)
        new_srcid_cnt += 1
        if new_srcid_cnt == math.floor(inc_num / 2):
            break
    new_srcid_cnt = 1
    for srcid, _ in sorted_min_values:
        if srcid in learning_srcids:
            continue
        the_cid = None
        for cid, cluster in cluster_dict.items():
            if srcid in cluster:
                the_cid = cid
                break
        if the_cid in added_cids:
            continue
        added_cids.append(the_cid)
        todo_srcids.append(srcid)
        new_srcid_cnt += 1
        if new_srcid_cnt == math.ceil(inc_num / 2):
            break
    return todo_srcids


def ir2tagset_al_query_samples_strict_phrase_util(test_srcids,
                                                  building,
                                                  sentence_dict,
                                                  pred_phrase_dict,
                                                  pred_tagsets_dict,
                                                  inc_num,
                                                  learning_srcids=[]):
    phrase_usage_dict = {}
    for srcid in test_srcids:
        if srcid in learning_srcids:
            continue
        pred_tagsets = pred_tagsets_dict[srcid]
        phrase_usage_dict[srcid] = determine_used_phrases(
                                       pred_phrase_dict[srcid],
                                       pred_tagsets)

    """
    phrase_usages = list(phrase_usage_dict.values())
    mean_usage_rate = np.mean(phrase_usages)
    std_usage_rate = np.std(phrase_usages)
    threshold = mean_usage_rate - std_usage_rate
    todo_sentence_dict = dict((srcid, alpha_tokenizer(''.join(
                               sentence_dict[srcid])))
                               for srcid, usage_rate
                               in phrase_usage_dict.items()
                               if usage_rate < threshold and srcid in test_srcids)
    """
    sorted_phrase_usages = sorted(phrase_usage_dict.items(), key=itemgetter(1))
    cluster_dict = get_cluster_dict(building)
    added_cids = []
    #TODO: Init added_cids from learning_srcids
    cand_srcids = [srcid for srcid in phrase_usage_dict.keys()
                   if srcid not in learning_srcids]


    todo_srcids = []
    new_srcid_cnt = 0
    for srcid in cand_srcids:
        the_cid = None
        for cid, cluster in cluster_dict.items():
            if srcid in cluster:
                the_cid = cid
                break
        if the_cid in added_cids:
            continue
        added_cids.append(the_cid)
        todo_srcids.append(srcid)
        new_srcid_cnt += 1
        if new_srcid_cnt == inc_num:
            break
    if new_srcid_cnt < inc_num:
        todo_srcids += select_random_samples(building,
                                            cand_srcids,
                                            inc_num - new_srcid_cnt,
                                            use_cluster_flag=True,
                                            token_type='justseparate',
                                            reverse=True,
                                            cluster_dict=None,
                                            shuffle_flag=True
                                            )
    return todo_srcids

def ir2tagset_al_query_samples_phrase_util(test_srcids,
                                           building,
                                           sentence_dict,
                                           pred_phrase_dict,
                                           pred_tagsets_dict,
                                           inc_num,
                                           learning_srcids=[]):
    phrase_usage_dict = {}
    for srcid in test_srcids:
        if srcid in learning_srcids:
            continue
        pred_tagsets = pred_tagsets_dict[srcid]
        phrase_usage_dict[srcid] = determine_used_phrases(
                                       pred_phrase_dict[srcid],
                                       pred_tagsets)

    phrase_usages = list(phrase_usage_dict.values())
    mean_usage_rate = np.mean(phrase_usages)
    std_usage_rate = np.std(phrase_usages)
    # Select underexploited sentences.
    threshold = mean_usage_rate - std_usage_rate
    todo_sentence_dict = dict((srcid, alpha_tokenizer(''.join(
                               sentence_dict[srcid])))
                               for srcid, usage_rate
                               in phrase_usage_dict.items()
                               if usage_rate < threshold and srcid in test_srcids)
    cluster_dict = get_cluster_dict(building)
    todo_srcids = select_random_samples(building, \
                          list(todo_sentence_dict.keys()),
                          min(inc_num, len(todo_sentence_dict)), \
                          True,\
                          reverse=True,
                          cluster_dict=cluster_dict,
                          shuffle_flag=False
                         )
    #if the numbers are not enough randomly select more:
    if len(todo_srcids) < inc_num:
        more_num = inc_num - len(todo_srcids)
        todo_sentence_dict = dict((srcid, alpha_tokenizer(''.join(
                                   sentence_dict[srcid])))
                                   for srcid, usage_rate
                                   in phrase_usage_dict.items()
                                   if srcid in test_srcids)
        cluster_dict = get_cluster_dict(building)
        todo_srcids = select_random_samples(building, \
                              list(todo_sentence_dict.keys()),
                              min(more_num, len(todo_sentence_dict)), \
                              True,\
                              cluster_dict=cluster_dict,
                              shuffle_flag=True
                             )
    return todo_srcids

def ir2tagset_al_query_samples_token_util(learning_srcids,
                                                 building,
                                                 sentence_dict,
                                                 token_label_dict,
                                                 pred_tagsets_dict,
                                                 inc_num):
    # Calculate utilization
    cand_srcids = [srcid for srcid in pred_tagsets_dict.keys()
                      if srcid not in learning_srcids]
    token_usage_dict = determine_used_tokens_multiple(\
                                sentence_dict, token_label_dict, \
                                pred_tagsets_dict, cand_srcids)
    token_usage_rate_dict = dict((srcid, sum(usage)/len(usage)) for 
                                 srcid, usage in token_usage_dict.items())
    usages = list(token_usage_rate_dict.values())
    mean_usage_rate = np.mean(usages)
    std_usage_rate = np.std(usages)
    threshold = mean_usage_rate - std_usage_rate
    todo_sentence_dict = dict((srcid, alpha_tokenizer(''.join(\
                                                sentence_dict[srcid]))) \
                              for srcid, usage_rate \
                              in token_usage_rate_dict.items()
                              if usage_rate<threshold and srcid in cand_srcids)

    todo_srcids = select_random_samples(building, \
                          list(todo_sentence_dict.keys()),
                          min(inc_num, len(todo_sentence_dict)), \
                          True,\
                          token_type='justseparate',
                          reverse=True,
                          cluster_dict=get_cluster_dict(building),
                          shuffle_flag=False
                         )
    return todo_srcids

def ir2tagset_al_query_entropy(target_prob_mat, 
                               target_srcids, 
                               learning_srcids,
                               target_building,
                               inc_num
                               ):
    entropies = get_entropy(target_prob_mat.T)
    sorted_entropies = sorted([(target_srcids[i], ent) for i, ent 
                               in enumerate(entropies)], key=itemgetter(1))
    cluster_dict = get_cluster_dict(target_building)
    added_cids = []
    todo_srcids = []
    new_srcid_cnt = 0
    for srcid, ent in sorted_entropies:
        if srcid in learning_srcids:
            continue
        the_cid = None
        for cid, cluster in cluster_dict.items():
            if srcid in cluster:
                the_cid = cid
                break
        if the_cid in added_cids:
            continue
        added_cids.append(the_cid)
        todo_srcids.append(srcid)
        new_srcid_cnt += 1
        if new_srcid_cnt == inc_num:
            break
    return todo_srcids



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


def determine_used_tokens(sentence, token_labels, tagsets):
    """
    Calculate how much tokens are used for the labels. == utilization
    """
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
    """
    Calculate utilization per srcid
    """

    token_usage_dict = dict()
    for srcid in srcids:
        token_usage_dict[srcid] = determine_used_tokens(\
                                        sentence_dict[srcid],\
                                        token_label_dict[srcid],\
                                        tagsets_dict[srcid])
    return token_usage_dict


def ir2tagset_onestep(step_data,
                     building_list,
                     source_sample_num_list,
                     target_building,
                     use_cluster_flag=False,
                     use_brick_flag=True,
                     eda_flag=False,
                     negative_flag = True,
                     debug_flag=False,
                     n_jobs=4,
                     ts_flag=False,
                     inc_num=5,
                     query_strategy='phrase_util'):
    
    #step_data['learning_srcids'] = step_data['next_learning_srcids']

    global tagset_list
    global total_srcid_dict
    global tagset_tree
    global tree_depth_dict
    
    ### Read CRF Data from step_data
    with open('metadata/{0}_label_dict_justseparate.json'
                  .format(target_building), 'r') as fp:
        srcids = json.load(fp).keys()
    # Read CRF result
    crf_phrase_dict = step_data['phrase_dict']
    learning_srcids = step_data['learning_srcids']
    crf_srcids = list(crf_phrase_dict.keys())

    # Get test data
    test_srcids = [srcid for srcid in srcids 
                   if srcid not in learning_srcids]

    test_sentence_dict,\
    test_token_label_dict,\
    test_phrase_dict,\
    test_truths_dict,\
    test_srcid_dict = get_building_data(target_building, test_srcids)

    ### Initialize Given (Learning) Data
    (crf_sentence_dict,
    crf_token_label_dict,
    _,
    crf_truths_dict, 
    _) = get_building_data(target_building, crf_srcids)

    given_sentence_dict, \
    given_token_label_dict, \
    given_truths_dict, \
    given_phrase_dict = get_multi_buildings_data(building_list, learning_srcids)

    # Add tagsets in the learning dataset if not exists.
    extend_tagset_list(reduce(adder, \
                [given_truths_dict[srcid] for srcid in learning_srcids]\
                + [crf_truths_dict[srcid] for srcid in crf_srcids], []))
    augment_tagset_tree(tagset_list) 
    source_target_buildings = list(set(building_list + [target_building]))

    # Learning IR->Tagset model
    classifier, vectorizer, binarizer, point_classifier, ts2ir = \
            build_tagset_classifier(building_list, target_building,\
                                    crf_sentence_dict, crf_token_label_dict,\
                                    given_phrase_dict, crf_phrase_dict,\
                                    given_truths_dict,\
                                    learning_srcids, crf_srcids,\
                                    tagset_list, eda_flag, use_brick_flag,
                                    source_target_buildings,
                                    n_jobs,
                                    ts_flag,
                                    negative_flag
                                   )
    tree_depth_dict = calc_leaves_depth(tagset_tree)

    # Infer Tagsets based on the above model.
    crf_pred_tagsets_dict, \
    crf_pred_certainty_dict, \
    crf_pred_point_dict,\
    target_prob_mat          = tagsets_prediction(\
                                   classifier, vectorizer, \
                                   binarizer, crf_phrase_dict, \
                                   crf_srcids, source_target_buildings,
                                   eda_flag, point_classifier, ts2ir)

    # Calculate utilization
    crf_token_usage_dict = determine_used_tokens_multiple(\
                                crf_sentence_dict, crf_token_label_dict, \
                                crf_pred_tagsets_dict, crf_srcids)
    crf_token_usage_rate_dict = dict((srcid, sum(usage)/len(usage))\
                                     for srcid, usage \
                                     in crf_token_usage_dict.items())
    usage_rates = list(crf_token_usage_rate_dict.values())
    usage_rate_mean = np.mean(usage_rates)
    usage_rate_mean = np.std(usage_rates)

    # Evaluate the result
    crf_entity_result_dict = tagsets_evaluation(crf_truths_dict, crf_pred_tagsets_dict,
                       crf_pred_certainty_dict, crf_srcids,
                       crf_pred_point_dict, crf_phrase_dict, debug_flag)

    #query_strategy = 'entropy'
    if query_strategy == 'token_util':
        new_srcids = ir2tagset_al_query_samples_token_util(
                         learning_srcids,
                         target_building,
                         crf_sentence_dict,
                         crf_token_label_dict,
                         crf_pred_tagsets_dict,
                         inc_num)
        step_data['next_learning_srcids'] += new_srcids
    elif query_strategy == 'phrase_util':
        new_srcids = ir2tagset_al_query_samples_phrase_util(
                            crf_srcids,
                            target_building,
                            crf_sentence_dict,
                            crf_phrase_dict,
                            crf_pred_tagsets_dict,
                            inc_num,
                            learning_srcids)
    elif query_strategy == 'strict_phrase_util':
        new_srcids = ir2tagset_al_query_samples_strict_phrase_util(
                            crf_srcids,
                            target_building,
                            crf_sentence_dict,
                            crf_phrase_dict,
                            crf_pred_tagsets_dict,
                            inc_num,
                            learning_srcids)
    elif query_strategy == 'entropy':
        new_srcids = ir2tagset_al_query_entropy(
                         target_prob_mat, 
                         crf_srcids, 
                         learning_srcids,
                         target_building,
                         inc_num)
    elif query_strategy == 'cmn':
        new_srcids = ir2tagset_al_query_samples_cmn(
                            target_prob_mat,
                            crf_srcids,
                            learning_srcids,
                            target_building)
    else:
        raise ValueError('Query Strategy Wrong: {0}'.format(query_strategy))

    # Select srcids to ask for the next step

    #todo_srcids = find_todo_srcids(crf_token_usage_rate_dict, crf_srcids, \
    #                               inc_num, crf_sentence_dict, target_building)
    #updated_learning_srcids = sorted(todo_srcids + curr_learning_srcids)

    next_step_data = deepcopy(step_data)
    next_step_data['next_learning_srcids'] += new_srcids
    if not next_step_data.get('result'):
        next_step_data['result'] = dict()
    if not next_step_data['result'].get('crf'):
        next_step_data['result']['crf'] = []
    if not next_step_data.get('learning_srcids_history'):
        next_step_data['learning_srcids_history'] = [learning_srcids]
    else:
        next_step_data['learning_srcids_history']
    next_step_data['result']['entity'] = crf_entity_result_dict
    return next_step_data


def ir2tagset_iteration(iter_num, custom_postfix, *args):
    """
    args: 
        building_list,
        source_sample_num_list,
        target_building,
        use_cluster_flag=False,
        use_brick_flag=False,
        crftype='crfsuite'
        learning_srcids=[]):
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
    step_datas = iteration_wrapper(iter_num, ir2tagset_onestep, prev_data, *args)

    building_list = args[0]
    target_building = args[2]
    postfix = 'ir2tagset_itr' 
    if custom_postfix:
        postfix += '_' + custom_postfix
    with open('result/crf_iter_{0}_{1}.json'\
            .format(''.join(building_list+[target_building]), postfix), 'w') as fp:
        json.dump(step_datas, fp, indent=2)
    end_time = arrow.get()
    print(iter_num, " iterations took: ", end_time - begin_time)


if __name__ == '__main__':
    tree_depth_dict = calc_leaves_depth(tagset_tree)
    print('max brick depth: {0}'.format(max(tree_depth_dict.values())))
