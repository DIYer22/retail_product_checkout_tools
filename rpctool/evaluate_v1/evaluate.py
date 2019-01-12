# Evaluation Code
# Prediction tpye: {filename_1: pred_1, filename_2: pred_2, ...}
# Label type:      {filename_1: label_1, filename_2: label_2, ...}
# Key type: str
# Value type: list of integers

import json
import numpy as np
from collections import defaultdict
from boxx import *

K = 200
def calculate(predictions, labels, log=True):
    # Inputs Example:
    # predictions = {'file1':[0,0,1], 'file2':[0,0,2], 'file3':[1,1,2], 'file4':[2,3,3]}
    # labels = {'file1':[0,0,1], 'file2':[0,1,1], 'file3':[1,0,2], 'file4':[0,0,2]}

    pred_keys, pred_values = list(predictions.keys()), np.array(list(predictions.values())).astype(np.float32)
    label_values  = np.array([ labels[key] for key in pred_keys ]).astype(np.float32)

    N = len(predictions)
    K = len(labels[pred_keys[0]])

    score_1 = score1(pred_values, label_values, N)
    score_2 = score2(pred_values, label_values, N)
    score_3 = score3(pred_values, label_values, N)
    score_4 = score4(pred_values, label_values, K)
    score_5 = score5(pred_values, label_values, K)    
    if log:
        print('Score1 is {:.4f}, Score2 is {:.4f}, Score3 is {:.4f}, Score4 is {:.4f}, Score5 is {:.4f}'.format(score_1, score_2, score_3, score_4, score_5))
    return dict(cAcc=score_1, ACD=score_2, score_3=score_3, mCCD=score_4, mCIoU=score_5)

def evaluate(pred_counts, gt_counts, log=True):
    predictions = defaultdict(lambda :[0]*K)
    labels = defaultdict(lambda :[0]*K)
    for k,vs in pred_counts.items():
        for v in vs:
            predictions[k][v] += 1
    for k,vs in gt_counts.items():
        for v in vs:
            labels[k][v] += 1
    return calculate(predictions, labels, log=log)

def score1(pred_values, label_values, N):
    return np.sum(np.all(np.equal(label_values, pred_values), axis=1)) / N


def score2(pred_values, label_values, N):
    return np.sum(np.abs(label_values - pred_values)) / N


def score3(pred_values, label_values, N):
    total_cd = np.sum(np.abs(label_values - pred_values), axis=1)
    total_gd = np.sum(label_values, axis=1)
    return np.sum(np.divide(total_cd, total_gd)) / N


def score4(pred_values, label_values, K):
    class_cd = np.sum(np.abs(label_values - pred_values), axis=0)
    class_gt = np.sum(label_values, axis=0)
    return np.sum(np.divide(class_cd, class_gt)) / K

def score5(pred_values, label_values, K):
    minimum = np.minimum(pred_values, label_values)
    maximum = np.maximum(pred_values, label_values)
    class_minimum = np.sum(minimum, axis=0)
    class_maximum = np.sum(maximum, axis=0)
    return np.sum(np.divide(class_minimum, class_maximum)) / K