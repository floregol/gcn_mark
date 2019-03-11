import matplotlib.pyplot as plt
import pickle as pk
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from utils import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from scipy.stats import entropy
from sklearn.preprocessing import normalize
import math

# result_path = 'results/classification/'
# plot_path = 'plots/classification/'

# adj, initial_features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = load_data('cora')

# onlyfiles = [f for f in listdir(result_path) if isfile(join(result_path, f))]

# average_softmax_results = np.zeros((2708, 7))
# y_bar = [0.14119789, 0.10435302, 0.16252837, 0.20183952, 0.17040856, 0.11477953, 0.10489289]
# for softmax_files in onlyfiles:
#     with open(result_path + softmax_files, 'rb') as f:
#         partial_softmax_results = pk.load(f, encoding='latin1')
#     average_softmax_results = average_softmax_results + partial_softmax_results

result_path = 'results/classification/classes/'

adj, initial_features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = load_data('cora')

onlyfiles = [f for f in listdir(result_path) if isfile(join(result_path, f))]

average_classes_results = np.zeros((2708, 7))
#y_bar = [0.14119789, 0.10435302, 0.16252837, 0.20183952, 0.17040856, 0.11477953, 0.10489289]
for classes_files in onlyfiles:
    with open(result_path + classes_files, 'rb') as f:
        partial_classes_results = pk.load(f, encoding='latin1')
    average_classes_results = average_classes_results + partial_classes_results

classified_nodes = np.argwhere(np.sum(average_classes_results, axis=1)).flatten()
label_ground_truth = np.argmax(labels[classified_nodes, :], axis=1)

y_bar = np.sum(labels, axis=0) / np.sum(labels)


def balance(v):
    if np.sum(v) > 0:
        # for i in range(v.shape[0]):
        #     v[i] = v[i] / y_bar[i]
        return v / np.sum(v)
    return v


# print(average_classes_results[classified_nodes, :][0:10])
average_classes_results = np.apply_along_axis(balance, 1, average_classes_results)
# print(average_classes_results[classified_nodes, :][0:10])
# print(labels[classified_nodes, :][0:10])
label_predictions = np.argmax(average_classes_results[classified_nodes, :], axis=1)

print(accuracy_score(label_predictions, label_ground_truth))
print(f1_score(label_predictions, label_ground_truth, average="macro"))

args_good = np.where(label_predictions == label_ground_truth)
args_not = np.where(label_predictions != label_ground_truth)
#print(label_predictions[args_not])
# print(label_ground_truth)

correct_softmax = average_classes_results[classified_nodes, :][args_good, :][0]

not_correct_softmax = average_classes_results[classified_nodes, :][args_not, :][0]

entropy_list = np.apply_along_axis(entropy, 1, correct_softmax)
entropy_list_bad = np.apply_along_axis(entropy, 1, not_correct_softmax)
max_entropy = math.log(7)
print(np.mean(entropy_list) / max_entropy)
print(np.mean(entropy_list_bad) / max_entropy)
# print(accuracy_score(label_ground_truth, label_predictions))
