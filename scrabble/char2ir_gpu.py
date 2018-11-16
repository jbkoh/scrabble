import os
from uuid import uuid4
from operator import itemgetter
from pathlib import Path

#import pycrfsuite
#from bson.binary import Binary as BsonBinary
import arrow
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, f1_score

import keras
from keras.layers import Embedding, Input, Masking
from keras.models import Model
from keras.backend.tensorflow_backend import set_session
from keras_contrib.layers.crf import CRF
from keras.models import Sequential
import tensorflow as tf

from mongo_models import store_model, get_model, get_tags_mapping, \
    get_crf_results, store_result, get_entity_results
from base_scrabble import BaseScrabble
from jasonhelper import bidict
from common import *
import eval_func

#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
#set_session(tf.Session(config=config))

curr_dir = Path(os.path.dirname(os.path.abspath(__file__)))

def gen_uuid():
    return str(uuid4())


class Char2Ir(BaseScrabble):
    def __init__(self,
                 target_building,
                 target_srcids,
                 building_label_dict,
                 building_sentence_dict,
                 source_buildings=[],
                 source_sample_num_list=[],
                 learning_srcids=[],
                 conf={}
                 ):
        super(Char2Ir, self).__init__(
                 target_building,
                 target_srcids,
                 building_label_dict,
                 building_sentence_dict,
                 {},
                 source_buildings,
                 source_sample_num_list,
                 learning_srcids,
                 conf)

        if 'query_strategy' in conf:
            self.query_strategy = conf['query_strategy']
        else:
            self.query_strategy = 'confidence'
        if 'user_cluster_flag' in conf:
            self.use_cluster_flag = conf['use_cluster_flag']
        else:
            self.use_cluster_flag = True

        # Model configuration for Keras
        self._config_keras()

        # Note: Hardcode to disable use_brick_flag
        """
        if 'use_brick_flag' in conf:
            self.use_brick_flag = conf['use_brick_flag']
        else:
            self.use_brick_flag = False  # Temporarily disable it
        """
        self.le = None
        self.feature_dict = bidict()
        self.use_brick_flag = False
        self._init_data()

    def _config_keras(self):
        self.epochs = 30
        self.batch_size = 16
        self.lr = 0.06
        #self.lr = 0.03
        self.opt = keras.optimizers.RMSprop(lr=self.lr)
        #self.opt = keras.optimizers.Adam(lr=self.lr)
        #self.opt = keras.optimizers.SGD(lr=self.lr, momentum=0.0,
        #                                decay=0.0, nesterov=False)
        #self.opt = keras.optimizers.Adagrad(lr=0.1, epsilon=None, decay=0.0)
        self.unroll_flag = False
        self.model = None

    def _init_data(self):
        self.sentence_dict = {}
        self.label_dict = {}
        learning_srcids = []
        self.degrade_mask = []
        for building, source_sample_num in zip(self.source_buildings,
                                               self.source_sample_num_list):
            self.sentence_dict.update(self.building_sentence_dict[building])
            one_label_dict = self.building_label_dict[building]
            self.label_dict.update(one_label_dict)

            if not self.learning_srcids:
                sample_srcid_list = select_random_samples(
                                        building,
                                        one_label_dict.keys(),
                                        source_sample_num,
                                        self.use_cluster_flag)
                learning_srcids += sample_srcid_list
                if building == self.target_building:
                    self.degrade_mask += [0] * len(sample_srcid_list)
                else:
                    self.degrade_mask += [1] * len(sample_srcid_list)
        if not self.learning_srcids:
            self.learning_srcids = learning_srcids
        self.max_len = max([len(sentence) for sentence
                            in self.sentence_dict.values()])

        # Construct Brick examples
        brick_sentence_dict = dict()
        brick_label_dict = dict()
        if self.use_brick_flag:
            with open(curr_dir / 'metadata/brick_tags_labels.json', 'r') as fp:
                tag_label_list = json.load(fp)
            for tag_labels in tag_label_list:
                # Append meaningless characters before and after the tag
                # to make it separate from dependencies.
                # But comment them out to check if it works.
                # char_tags = [' '] + list(map(itemgetter(0), tag_labels)) + [' ']
                char_tags = list(map(itemgetter(0), tag_labels))
                # char_labels = ['O'] + list(map(itemgetter(1), tag_labels)) + ['O']
                char_labels = list(map(itemgetter(1), tag_labels))
                brick_sentence_dict[''.join(char_tags)] = char_tags + ['NEWLINE']
                brick_label_dict[''.join(char_tags)] = char_labels + ['O']
            self.sentence_dict.update(brick_sentence_dict)
            self.label_dict.update(brick_label_dict)
        self.brick_srcids = list(brick_sentence_dict.keys())

    def encode_labels(self, label_dict, srcids):
        if not self.le:
            with open('brick/tags.json', 'r') as fp:
                brick_tags = json.load(fp)
            #flat_labels = reduce(adder, [label_dict[srcid] for srcid in srcids])
            flat_labels = ['B_' + tag for tag in brick_tags] + \
                          ['I_' + tag for tag in brick_tags] + \
                          ['O'] + \
                          reduce(adder, [label_dict[srcid] for srcid in srcids])
            self.le = LabelBinarizer().fit(flat_labels)
        stack = []
        for srcid in srcids:
            labels = label_dict[srcid]
            encoded = self.le.transform(labels)
            encoded = np.vstack([encoded, np.zeros(
                                 (self.max_len - encoded.shape[0],
                                  encoded.shape[1]))])
            stack.append(encoded)
        return np.stack(stack)

    def encode_labels_dep(self, label_dict, srcids):
        flat_labels = reduce(adder, [label_dict[srcid] for srcid in srcids])
        self.le = LabelBinarizer().fit(flat_labels)
        stack = []
        for srcid in srcids:
            labels = label_dict[srcid]
            encoded = self.le.transform(labels)
            encoded = np.vstack([encoded, np.zeros(
                                 (self.max_len - encoded.shape[0],
                                  encoded.shape[1]))])
            stack.append(encoded)
        return np.stack(stack)

    def _weight_logic(self, features, degrade_mask):
        sample_size = features.shape[0]
        weights = np.ones(sample_size)
        external_sample_size = sum(degrade_mask)
        weights -= np.multiply(np.ones(sample_size)
                               * external_sample_size / sample_size,
                               degrade_mask)
        return weights

    def learn_model(self, features, labels, degrade_mask, epochs=10, batch_size=None,
                    model=None):
        if not model and not self.model:
            model = Sequential()
            masking = Masking(mask_value=0.0, input_shape=(features.shape[1], features.shape[2],))
            model.add(masking)
            crf = CRF(#input_shape=(features.shape[1], features.shape[2],),
                      units=labels.shape[-1],
                      sparse_target=False,
                      #kernel_regularizer=keras.regularizers.l2(0.),
                      #bias_regularizer=keras.regularizers.l2(0.005),
                      #chain_regularizer=keras.regularizers.l2(0.005),
                      #boundary_regularizer=keras.regularizers.l2(0.005),
                      learn_mode='marginal',
                      test_mode='marginal',
                      unroll=self.unroll_flag,
                     )
            model.add(crf)
            model.compile(optimizer=self.opt,
                          loss=crf.loss_function,
                          metrics=[crf.accuracy])
        elif self.model:
            model = self.model

        #assert features.shape[0] == len(self.degrade_mask)
        weights = self._weight_logic(features, degrade_mask)

        model.fit(features, labels, epochs=epochs, batch_size=batch_size,
                       verbose=1, sample_weight=weights)
        return model

    def update_feature_dict(self, sentences):
        features = set(['BOS', 'isdigit'])#, 'SECOND', 'LAST'])
        for sentence in sentences:
            sentence_len = len(sentence)
            for i, word in enumerate(sentence):
                features.add(word.lower())
                if i > 1:
                    features.add('-1:' + sentence[i-1])
                #if i > 2:
                #    features.add('-2:' + sentence[i-2])
                #if i < sentence_len - 1:
                #    features.add('+1:' + sentence[i+1])
        for i, feat_type in enumerate(features):
            self.feature_dict[feat_type] = i

    def featurize(self, sentence_dict, srcids):
        feat_list = [self._calc_feature(sentence_dict[srcid], self.max_len) 
                     for srcid in srcids]
        features = np.vstack(feat_list)
        return features

    def _calc_feature(self, sentence, max_len):
        feature = np.zeros((1, max_len, len(self.feature_dict)))
        sentence_len = len(sentence)
        for i, word in enumerate(sentence):
            feats = []
            if i == 0:
                feats.append('BOS')
            #if i == 1:
            #    feats.append('SECOND')
            #if i == sentence_len:
            #    feats.append('LAST')
            if word == 'number':
                feats.append('isdigit')
            feats.append(word.lower())
            if i > 0:
                feats.append('-1:' + sentence[i-1].lower())
            #if i > 1:
            #    feats.append('-2:' + sentence[i-2].lower())
            #if i < sentence_len - 1:
            #    feats.append('+1:' + sentence[i+1].lower())

            for feat in feats:
                if feat in self.feature_dict:
                    feature [0, i, self.feature_dict[feat]] = 1
                else:
                    print('Feature "{0}" is not initiated'.format(feat))

        return feature

    def update_model(self, srcids, building=None):
        if not building:
            building = self.target_building
        if building == self.target_building:
            #self.degrade_mask += [0] * len(srcids)
            curr_degrade_mask = [0] * len(srcids)
        assert (len(self.source_buildings) == len(self.source_sample_num_list))
        #self.learning_srcids += srcids * 5
        self.learning_srcids += srcids
        #TODO: If needed, internalize variables inside those functions.
        self.update_feature_dict(self.sentence_dict.values())
        #if not srcids:
        if True:
            train_features = self.featurize(self.sentence_dict, self.learning_srcids)
            train_labels = self.encode_labels(self.label_dict, self.learning_srcids)
            degrade_mask = self.degrade_mask + curr_degrade_mask
            self.degrade_mask = degrade_mask
        else:
            train_features = self.featurize(self.sentence_dict, srcids)
            train_labels = self.encode_labels(self.label_dict, srcids)
            degrade_mask = curr_degrade_mask
        self.model = self.learn_model(train_features,
                                      train_labels,
                                      degrade_mask,
                                      epochs=self.epochs,
                                      batch_size=self.batch_size)
        model_metadata = {
            # 'source_list': sample_dict,
            'gen_time': arrow.get().datetime,
            'use_cluster_flag': self.use_cluster_flag,
            'use_brick_flag': self.use_brick_flag,
            #'model_file': model_file,
            'source_building_count': len(self.source_buildings),
            'learning_srcids': sorted(set(self.learning_srcids)),
            #'uuid': self.model_uuid,
            'crftype': 'crfsuite'
        }
        store_model(model_metadata)

    def select_informative_samples(self, sample_num):
        target_srcids = [srcid for srcid in self.target_srcids
                         if srcid not in self.learning_srcids]
        target_sentence_dict = {srcid: self.sentence_dict[srcid]
                                for srcid in target_srcids}

        predicted_dict, score_dict, pred_phrase_dict = \
            self._predict_and_proba(target_srcids)
        cluster_dict = get_cluster_dict(self.target_building)

        new_srcids = []
        if self.query_strategy == 'confidence':
            for srcid, score in score_dict.items():
                #Normalize with length
                score_dict[srcid] = score / len(self.sentence_dict[srcid])
            sorted_scores = sorted(score_dict.items(), key=itemgetter(1))

            # load word clusters not to select too similar samples.
            added_cids = []
            new_srcid_cnt = 0
            for srcid, score in sorted_scores:
                if srcid in target_srcids:
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
                    if new_srcid_cnt == sample_num:
                        break
        return new_srcids

    def _calc_features(self, sentence, building=None):
        sentenceFeatures = list()
        sentence = ['$' if c.isdigit() else c for c in sentence]
        for i, word in enumerate(sentence):
            features = {
                'word.lower=' + word.lower(): 1.0,
                'word.isdigit': float(word.isdigit())
            }
            if i == 0:
                features['BOS'] = 1.0
            else:
                features['-1:word.lower=' + sentence[i - 1].lower()] = 1.0
            if i in [0, 1]:
                features['SECOND'] = 1.0
            else:
                features['-2:word.lower=' + sentence[i - 2].lower()] = 1.0
            # if i<len(sentence)-1:
            #    features['+1:word.lower='+sentence[i+1].lower()] = 1.0
            # else:
            #    features['EOS'] = 1.0
            sentenceFeatures.append(features)
        return sentenceFeatures

    def decode_labels(self, preds, target_srcids):
        pred_labels = {}
        for pred, srcid in zip(preds, target_srcids):
            try:
                decoded = self.le.inverse_transform(pred)
                sentence_len = len(self.sentence_dict[srcid])
            except:
                pdb.set_trace()
            pred_labels[srcid] = decoded[:sentence_len].tolist()
        return pred_labels

    def _predict_and_proba(self, target_srcids, score_flag=True, model=None):
        # Validate if we have all information
        if not model:
            assert self.model
            model = self.model
        for srcid in target_srcids:
            assert srcid in self.sentence_dict

        features = self.featurize(self.sentence_dict, target_srcids)
        preds = model.predict(features)
        if score_flag:
            np.amax(np.log(preds), axis=2)
            begin_time = arrow.get()
            marginal_log_probs = np.amax(np.log(preds), axis=2)
            scores = {}
            for i, srcid in enumerate(target_srcids):
                sent_len = len(self.sentence_dict[srcid])
                scores[srcid] = np.sum(marginal_log_probs[i][0:sent_len])
            #scores = {srcid: self.model.test_on_batch(
            #                     features[i:i+1], preds[i:i+1])[0]
            #          for i, srcid
            #          in zip(range(0, features.shape[0]), target_srcids)}
            end_time = arrow.get()
            #pdb.set_trace()
        else:
            scores = {}
        pred_labels = self.decode_labels(preds, target_srcids)
        # Construct output data
        target_sentence_dict = {srcid: self.sentence_dict[srcid]
                                for srcid in target_srcids}
        pred_phrase_dict = make_phrase_dict(target_sentence_dict, pred_labels)
        return pred_labels, scores, pred_phrase_dict

    def predict(self, target_srcids=None):
        if not target_srcids:
            target_srcids = self.target_srcids
        predicted_dict, _, _ = self._predict_and_proba(target_srcids, False)
        self.predicted_dict = predicted_dict
        return predicted_dict

    def predict_proba(self, target_srcids=None):
        if not target_srcids:
            target_srcids = self.target_srcids
        _, score_dict, _ = self._predict_and_proba(target_srcids)
        return score_dict

    def learn_auto(self, iter_num=1):
        pass

    def comp_res(self, srcid):
        comp_true_pred(self.label_dict[srcid],
                       self.predicted_dict[srcid],
                       self.sentence_dict[srcid])

    def evaluate(self, preds):
        acc = eval_func.sequential_accuracy(
            [self.label_dict[srcid] for srcid in preds.keys()],
            [preds[srcid] for srcid in preds.keys()])
        pred = [preds[srcid] for srcid in preds.keys()]
        true = [self.label_dict[srcid] for srcid in preds.keys()]
        mlb = MultiLabelBinarizer()
        mlb.fit(pred + true)
        encoded_true = mlb.transform(true)
        encoded_pred = mlb.transform(pred)
        macro_f1 = f1_score(encoded_true, encoded_pred, average='macro')
        f1 = f1_score(encoded_true, encoded_pred, average='weighted')
        res = {
            'accuracy': acc,
            'f1': f1,
            'macro_f1': macro_f1
        }
        return res

