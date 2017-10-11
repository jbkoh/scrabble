from copy import deepcopy
from sklearn.metrics import f1_score

import numpy as np

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

def get_macro_f1(true_mat, pred_mat):
    assert true_mat.shape == pred_mat.shape
    f1s = []
    for i in range(0, true_mat.shape[1]):
        if 1 not in true_mat[:,i]:
            continue
        f1 = f1_score(true_mat[:,i], pred_mat[:,i])
        f1s.append(f1)
    return np.mean(f1s)


def get_accuracy(true_mat, pred_mat):
    acc_list = list()
    for true, pred in zip(true_mat, pred_mat):
        true_pos_indices = set(np.where(true==1)[0])
        pred_pos_indices = set(np.where(pred==1)[0])
        acc = len(pred_pos_indices.intersection(true_pos_indices)) /\
                len(pred_pos_indices.union(true_pos_indices))
        acc_list.append(acc)
    return np.mean(acc_list)
