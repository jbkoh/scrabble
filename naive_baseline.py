import json
import pdb
import re
import argparse
from operator import itemgetter
from multiprocessing import Pool
from functools import partial
from copy import deepcopy
import os

import numpy as np
from skmultilearn.problem_transform import LabelPowerset, ClassifierChain, \
                                           BinaryRelevance

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support

from scrabble_hierarchy import select_random_samples, get_micro_f1, get_accuracy
import building_tokenizer as toker

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
parser = argparse.ArgumentParser()
parser.register('type','bool',str2bool)
parser.register('type','slist', str2slist)
parser.register('type','ilist', str2ilist)
parser.add_argument('-bl',
                    type='slist',
                    help='Learning source building name list',
                    dest='building_list')
parser.add_argument('-nl',
                    type='ilist',
                    help='A list of the number of learning sample',
                    dest='sample_num_list')
parser.add_argument('-t',
                    type=str,
                    help='Target buildling name',
                    dest='target_building')
parser.add_argument('-avg',
                    type=int,
                    help='Number of exp to get avg. If 1, ran once',
                    dest='avgnum',
                    default=1)
args = parser.parse_args()

def replacer(s):
    if re.findall('\d+', s):
        return ' '
        #return '0'
    elif re.findall('[^0-9a-zA-Z]+', s):
        #return 'specialcharacter'
        return ' '
    else:
        return s.lower()

def joiner(x):
    return ' '.join(x)

def tokenizer(x):
    return x.split()

#building_list = ['ebu3b', 'ap_m']
#target_building = 'ap_m'
#n_list = [200, 10]
building_list = args.building_list
n_list = args.sample_num_list
target_building = args.target_building
avg_num = args.avgnum

iter_num = 10
n_jobs = 6
accuracy_list_list = list()
micro_f1_list_list = list()

inc_num = 5

def naive_base(params):
    building_list = params[0]
    n_list = params[1]
    target_building = params[2]
    accuracy_list = list()
    micro_f1_list = list()
    macro_f1_list = list()
    for iter_i in range(0, iter_num):
        sentence_dict = dict()
        truth_dict = dict()
        learning_srcids = list()
        for building, n in zip(building_list, n_list):
            if building == target_building:
                n += iter_i * inc_num
            if building != 'ghc':
                sensorDF, srcid_list, name_list, jciname_list, desc_list, unit_list, bacnettype_list =\
                        toker.parse_sentences(building)
                for srcid, name, jciname, desc in zip(srcid_list, name_list, jciname_list, desc_list):
                    sentence_dict[srcid] = list(map(replacer, name + jciname + desc))
            else:
                with open('metadata/{0}_sentence_dict_justseparate.json'.format(building), 'r') as fp:
                    curr_sentence_dict = json.load(fp)

                curr_sentence_dict = dict([(srcid, list(map(replacer, sentence))) for srcid, sentence in curr_sentence_dict.items()])
                sentence_dict.update(curr_sentence_dict)

            with open('metadata/{0}_ground_truth.json'.format(building), 'r') as fp:
                truth_dict.update(json.load(fp))
            srcids = list(truth_dict.keys())

            learning_srcids += select_random_samples(building,
                                                     srcids,\
                                                     n, \
                                                     True,\
                                                     token_type='justseparate',
                                                     reverse=True,
                                                     cluster_dict=None,
                                                     shuffle_flag=False
                                                    )
            if building == target_building:
                test_srcids = [srcid for srcid in srcids if srcid not in learning_srcids]

        binarizer = MultiLabelBinarizer().fit(truth_dict.values())
        vectorizer = TfidfVectorizer(tokenizer=tokenizer).fit(list(map(joiner, sentence_dict.values())))
        learning_doc = [' '.join(sentence_dict[srcid]) for srcid in learning_srcids]
        learning_vect_doc = vectorizer.transform(learning_doc)

        learning_truth_mat = binarizer.transform([truth_dict[srcid] for srcid in learning_srcids])

        #classifier = RandomForestClassifier(n_estimators=200, n_jobs=1)
        classifier = ClassifierChain(RandomForestClassifier())
        classifier.fit(learning_vect_doc, learning_truth_mat)

        test_doc = [' '.join(sentence_dict[srcid]) for srcid in test_srcids]
        test_vect_doc = vectorizer.transform(test_doc)

        pred_mat = classifier.predict(test_vect_doc)
        pred_tagsets_list = binarizer.inverse_transform(pred_mat)
        pred_tagsets_dict = dict([(srcid, pred_tagset) for srcid, pred_tagset in zip(test_srcids, pred_tagsets_list)])

        correct_cnt = 0
        incorrect_cnt = 0
        for i, srcid in enumerate(test_srcids):
            pred = pred_tagsets_dict[srcid]
            true = truth_dict[srcid]
            if set(pred_tagsets_dict[srcid]) != set(truth_dict[srcid]):
                incorrect_cnt += 1
            else:
                correct_cnt += 1

        test_truth_mat = binarizer.transform([truth_dict[srcid] for srcid in test_srcids])

        """
        prec_list = list()
        rec_list = list()
        f1_list = list()
        for i in range(0,test_truth_mat.shape[1]):
            true = test_truth_mat[:,i]
            pred = pred_mat.toarray()[:,i]
            prec, rec, f1, _ = precision_recall_fscore_support(true, pred)
            if np.sum(true)!=0 or np.sum(pred)!=0:
                prec_list.append(prec)
                rec_list.append(rec)
                f1_list.append(f1)
        """
        if not isinstance(pred_mat, np.ndarray):
            pred_mat = pred_mat.toarray()
        if not isinstance(test_truth_mat, np.ndarray):
            test_truth_mat = test_truth_mat.toarray()

        accuracy = get_accuracy(test_truth_mat, pred_mat)
        micro_f1 = get_micro_f1(test_truth_mat, pred_mat)
        _, _, macro_f1, _ = precision_recall_fscore_support(test_truth_mat,
                                                    pred_mat, average='macro')
        accuracy_list.append(accuracy * 100)
        micro_f1_list.append(micro_f1 * 100)
        macro_f1_list.append(macro_f1 * 100)

        #print('avg prec: {0}'.format(np.mean(prec_list)))
        #print('avg rec: {0}'.format(np.mean(rec_list)))
    return accuracy_list, macro_f1_list
    #micro_f1_list_list.append(micro_f1_list)
    #accuracy_list_list.append(accuracy_list)

p = Pool(n_jobs)
#mapped_naive = partial(naive_base, building_list, n_list, target_building)
params = [(deepcopy(building_list), deepcopy(n_list), deepcopy(target_building)) 
          for i in range(0, n_jobs)]
#params = [deepcopy((naive_base, building_list, n_list, target_building)) for i in range(0, n_jobs)]
result_list_list = p.map(naive_base, params)
p.close()
accuracy_list_list = list(map(itemgetter(0), result_list_list))
mf1_list_list = list(map(itemgetter(1), result_list_list))

acc_avg_list = [np.mean(list(map(itemgetter(i), accuracy_list_list))) \
                for i in range(0, iter_num)]
acc_std_list = [np.std(list(map(itemgetter(i), accuracy_list_list))) \
                for i in range(0, iter_num)]

mf1_avg_list = [np.mean(list(map(itemgetter(i), mf1_list_list))) \
                for i in range(0, iter_num)]
mf1_std_list = [np.std(list(map(itemgetter(i), mf1_list_list))) \
                for i in range(0, iter_num)]

print('Avg Accuracy: {0}'.format(acc_avg_list))
print('Std Accuracy: {0}'.format(acc_std_list))

result_file = 'result/baseline.json'
if os.path.isfile(result_file):
    with open(result_file, 'r') as fp:
        result_dict = json.load(fp)
else:
    result_dict = dict()
begin_num = n_list[-1]
result_dict[str(tuple(building_list))] = {
    'ns': n_list,
    'sample_numbers': list(range(begin_num, begin_num + inc_num * iter_num,
                                 inc_num)),
    'avg_acc': acc_avg_list,
    'std_acc': acc_std_list,
    'avg_mf1': mf1_avg_list,
    'std_mf1': mf1_std_list
}
with open(result_file, 'w') as fp:
    json.dump(result_dict ,fp, indent=2)

