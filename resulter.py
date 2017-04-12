import json
from collections import defaultdict, OrderedDict
from copy import copy

from pymongo import MongoClient
import arrow

from mongo_models import summary_query_template


class Resulter():
    def __init__(self, token_type="BILOU", spec={}):
        self.db = MongoClient().get_database('scrabble')
        self.result_coll = self.db.get_collection('results')
        self.summary_coll = self.db.get_collection('summary')
        self.result_dict = defaultdict(dict)
        self.summary = dict()
        self.token_type = token_type
        self.spec = spec

    def add_one_result(self, srcid, sentence, \
                             pred_token_labels, orig_token_labels):
        # Check the size of data
        assert(len(sentence)==len(pred_token_labels))
        assert(len(orig_token_labels)==len(pred_token_labels))

        # Add raw results
        self.result_dict[srcid] = {
                'sentence': sentence,
                'pred_token_labels': pred_token_labels,
                'orig_token_labels': orig_token_labels
                }

        # Gen labels per phrase
        pred_phrase_labels = list()
        if self.token_type=='BILOU':
            phraser = self._bilou_phraser
        self.result_dict[srcid]['pred_phrase_labels'] \
                = phraser(pred_token_labels)
        self.result_dict[srcid]['orig_phrase_labels'] \
                = phraser(orig_token_labels)

    def measure_accuracy_by_phrase(self, 
                                   pred_phrase_labels, 
                                   orig_phrase_labels,
                                   pessimistic_flag=False):
        #total_label_num = len(orig_phrase_labels)
        found_label_num = len(pred_phrase_labels)
        correct_label_num = 0
        total_label_num = 0
        for label in orig_phrase_labels:
            if pessimistic_flag and label in ['left_identifier', 
                                             'right_identifier', 
                                             'room', 
                                             'building']:
                continue
            total_label_num += 1
            if label in pred_phrase_labels:
                correct_label_num += 1
                pred_phrase_labels.remove(label)
        incorrect_label_num = total_label_num - correct_label_num
        return found_label_num, correct_label_num, incorrect_label_num
    
    def measure_accuracy_by_token(self, pred_token_labels, orig_token_labels):
        assert(len(pred_token_labels)==len(orig_token_labels))
        correct_cnt = 0
        incorrect_cnt = 0
        for pred, orig in zip(pred_token_labels, orig_token_labels):
            if pred==orig:
                correct_cnt += 1
            else:
                incorrect_cnt += 1
        return correct_cnt, incorrect_cnt

    def summarize_result(self):

        self.summary = {'specification': self.spec}
        # Calculate character level accuracy
        char_correct_cnt = 0
        char_total_cnt = 0

        correct_phrase_cnt = 0
        incorrect_phrase_cnt = 0
        predicted_phrase_cnt = 0

        for srcid, result in self.result_dict.items():
            correct_cnt, incorrect_cnt = self.measure_accuracy_by_token(\
                                            result['pred_token_labels'],\
                                            result['orig_token_labels'])
            char_correct_cnt += correct_cnt
            char_total_cnt += (correct_cnt + incorrect_cnt)
            found, correct, incorrect  = self.measure_accuracy_by_phrase(
                                            result['pred_phrase_labels'],
                                            result['orig_phrase_labels'])
            correct_phrase_cnt += correct
            incorrect_phrase_cnt += incorrect
            predicted_phrase_cnt += found
            
            pess_found, pess_correct, pess_incorrect  = self.measure_accuracy_by_phrase(
                                            result['pred_phrase_labels'],
                                            result['orig_phrase_labels'])
            correct_phrase_cnt += correct
            incorrect_phrase_cnt += incorrect
            predicted_phrase_cnt += found

        self.summary['char_precision'] = \
                                float(char_correct_cnt)/char_total_cnt
        self.summary['phrase_precision'] = \
        float(correct_phrase_cnt) / (correct_phrase_cnt + incorrect_phrase_cnt)
        self.summary['phrase_recall'] = \
        float(correct_phrase_cnt) / (predicted_phrase_cnt)
        self.summary['date'] = str(arrow.get().datetime)
        
    
    def serialize_summary(self, filename):
        with open(filename, 'w') as fp:
            json.dump(self.summary, fp, indent=2)


    def serialize_result(self, filename):
        with open(filename, 'w') as fp:
            json.dump(self.result_dict, fp, indent=2)

    def store_result_db(self):
        spec = self.spec
        summary_query = copy(summary_query_template)
        summary_query['source_building'] = spec['source_building']
        summary_query['target_building'] = spec['target_building']
        summary_query['source_sample_num'] = spec['source_sample_num']
        summary_query['label_type'] = spec['label_type']
        summary_query['token_type'] = spec['token_type']
        summary_query['use_cluster_flag'] = spec['use_cluster_flag']
        doc = copy(summary_query)
        doc.update(summary = self.summary)
        self.summary_coll.update(summary_query, 
                                 doc,
                                 upsert=True)

    def _bilou_phraser(self, token_labels):
        phrase_labels = list()
        curr_phrase = ''
        for i, label in enumerate(token_labels):
            tag = label[0] 
            if tag=='B':
                if curr_phrase: 
                # Below is redundant if the other tags handles correctly.
                    phrase_labels.append(curr_phrase)
                curr_phrase = label[2:] 
            elif tag=='I':
                if curr_phrase != label[2:]:
                    phrase_labels.append(curr_phrase)
                    curr_phrase = label[2:]
            elif tag=='L':
                if curr_phrase != label[2:]:
                    # Add if the previous label is different
                    phrase_labels.append(curr_phrase)
                # Add current label
                phrase_labels.append(label[2:])
                curr_phrase = ''
            elif tag=='O':
                # Do nothing other than pushing the previous label
                if curr_phrase: 
                    phrase_labels.append(curr_phrase)
                curr_phrase = ''
            elif tag=='U':
                if curr_phrase: 
                    phrase_labels.append(curr_phrase)
                phrase_labels.append(label[2:])
            else:
                print('Tag is incorrect in: {0}.'.format(label))
                assert(False)
        return phrase_labels

