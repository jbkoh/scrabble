import json
import random
from functools import reduce, partial
import logging
import re
from collections import defaultdict, OrderedDict
import pdb
import sys

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as hier

def sub_dict_by_key_set(d, ks):
    return dict((k,v) for k, v in d.items() if k in ks)

def leave_one_word(s, w):
    if w in s:
        s = s.replace(w, '')
        s = w + '-' + s
    return s

def find_keys(tv, d, crit):
    keys = list()
    for k, v in d.items():
        if crit(tv, v):
            keys.append(k)
    return keys

def check_in(x,y):
    return x in y

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
        sample_srcids = random.sample(srcids, n)
    return list(sample_srcids)

def splitter(s):
    return s.split('_')

def alpha_tokenizer(s): 
    return re.findall('[a-zA-Z]+', s)

def adder(x, y):
    return x+y

def bilou_tagset_phraser(sentence, token_labels):
    phrase_labels = list()
    curr_phrase = ''
    for i, (c, label) in enumerate(zip(sentence, token_labels)):
        if label[2:] in ['rightidentifier', 'leftidentifier']:
            continue
        tag = label[0]
        if tag=='B':
            if curr_phrase:
            # Below is redundant if the other tags handles correctly.       
                phrase_labels.append(curr_phrase)
            curr_phrase = label[2:]
        elif tag == 'I':
            if curr_phrase != label[2:] and curr_phrase:
                phrase_labels.append(curr_phrase)
                curr_phrase = label[2:]
        elif tag == 'L':
            if curr_phrase != label[2:] and curr_phrase:
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
            pdb.set_trace()
        if len(phrase_labels)>0:
            if phrase_labels[-1] == '':
                pdb.set_trace()
    if curr_phrase != '':
        phrase_labels.append(curr_phrase)
    phrase_labels = [leave_one_word(\
                         leave_one_word(phrase_label, 'leftidentifier'),\
                            'rightidentifier')\
                        for phrase_label in phrase_labels]
    phrase_labels = list(reduce(adder, map(splitter, phrase_labels), []))
    return phrase_labels

def make_phrase_dict(sentence_dict, token_label_dict):
    #phrase_dict = OrderedDict()
    phrase_dict = dict()
    for srcid, token_labels in token_label_dict.items():
        sentence = sentence_dict[srcid]
        phrases = bilou_tagset_phraser(sentence, token_labels)
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
        #phrase_dict[srcid] = phrases + phrases # TODO: Why did I put this before?
        phrase_dict[srcid] = phrases
    return phrase_dict

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
    value_lengther = lambda x: len(x[1])
    return OrderedDict(\
               sorted(cluster_dict.items(), key=value_lengther, reverse=True))

def set_logger(logfile=None):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s')

    # Console Handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    # File Handler
    if logfile:
        fh = logging.FileHandler(logfile, mode='w+')
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    return logger


def parallel_func(orig_func, return_idx, return_dict, *args):
    return_dict[return_idx] = orig_func(*args)


def iteration_wrapper(iter_num, func, prev_data=None, *params):
    step_datas = list()
    if not prev_data:
        prev_data = {'iter_num':0,
                     'learning_srcids': [],
                     'model_uuid': None}
    for i in range(0, iter_num):
        print('{0} th stage started'.format(prev_data['iter_num']))
        step_data = func(prev_data, *params)
        print('{0} th stage finished'.format(prev_data['iter_num']))
        step_datas.append(step_data)
        prev_data = step_data
        prev_data['iter_num'] += 1
    return step_datas


def replace_num_or_special(word):
    if re.match('\d+', word):
        return 'NUMBER'
    elif re.match('[a-zA-Z]+', word):
        return word
    else:
        return 'SPECIAL'

def adder(x, y):
    return x + y


def get_random_srcids(building_list, source_sample_num_list):
    srcids = list()
    for building, source_sample_num in zip(building_list, 
                                           source_sample_num_list): 
                                        #  Load raw sentences of a building
        with open("metadata/%s_char_sentence_dict.json" % (building), 
                      "r") as fp:
            sentence_dict = json.load(fp)
        sentence_dict = dict((srcid, [char for char in sentence]) 
                             for (srcid, sentence) in sentence_dict.items())

        # Load character label mappings.
        with open('metadata/{0}_char_label_dict.json'.format(building), 
                      'r') as fp:
            label_dict = json.load(fp)

        # Select learning samples.
        # Learning samples can be chosen from the previous stage.
        # Or randomly selected.
        sample_srcid_list = select_random_samples(building, \
                                                  label_dict.keys(), \
                                                  source_sample_num, \
                                                  use_cluster_flag=True)
        srcids += sample_srcid_list
    return srcids

def get_cluster_dict(building):
    cluster_filename = 'model/%s_word_clustering_justseparate.json' % \
                           (building)
    with open(cluster_filename, 'r') as fp:
        cluster_dict = json.load(fp)
    return cluster_dict

def get_label_dict(building):
    filename = 'metadata/%s_label_dict_justseparate.json' % \
                           (building)
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data
