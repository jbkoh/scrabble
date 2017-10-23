import json
import pdb
from functools import reduce
from collections import OrderedDict, Counter
import random

import re

def replace_num_or_special(word):
    if re.match('\d+', word):
        return 'NUMBER'
    elif re.match('[a-zA-Z]+', word):
        return word
    else:
        return 'SPECIAL'

building = 'ebu3b'
with open('metadata/{0}_sentence_dict_justseparate.json'.format(building), 
          'r') as fp:
    sentence_dict = json.load(fp)
srcids = list(sentence_dict.keys())
for srcid, sentence in sentence_dict.items():
    sentence_dict[srcid] = list(map(replace_num_or_special, sentence))

adder = lambda x,y: x + y

total_words = set(reduce(adder, sentence_dict.values()))
word_counter = Counter(reduce(adder, sentence_dict.values()))


with open('model/{0}_word_clustering_justseparate.json'.format(building), 
          'r') as fp:
    cluster_dict = json.load(fp)

# Learning Sample Selection
sample_srcids = set()
length_counter = lambda x: len(x[1])
ander = lambda x, y: x and y
n = 100
sample_cnt = 0
shuffle_flag = False
sorted_cluster_dict = OrderedDict(
    sorted(cluster_dict.items(), key=length_counter, reverse=True))
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

sample_sentence_dict = dict((srcid, sentence_dict[srcid])
                            for srcid in sample_srcids)

pdb.set_trace()
