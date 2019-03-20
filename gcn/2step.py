import random
import time
import tensorflow as tf

from utils import *
from models import GCN, MLP
import os
from scipy import sparse
from train import get_trained_gcn
from copy import copy, deepcopy
import pickle as pk
import multiprocessing as mp
import math
import sys
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import math
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
"""

 Moving the nodes around experiment

"""
result_path = 'results/classification/'
plot_path = 'plots/classification/'
intitial_gcn_neighbors_files = 'initial_neigh.pk'
intitial_gcn_files = 'initial.pk'
classifier_avg_softmax_file = 'classifier_avg_softmax_file.pk'
classifier_avg_wei_softmax_file = 'classifier_avg_wei_softmax_file.pk'
# Load result files
onlyfiles = [f for f in listdir(result_path) if isfile(join(result_path, f))]
onlyfiles.remove(intitial_gcn_neighbors_files)
onlyfiles.remove(intitial_gcn_files)
onlyfiles.remove(classifier_avg_softmax_file)
onlyfiles.remove(classifier_avg_wei_softmax_file)
dataset = 'cora'
adj, initial_features,y_train, y_val, y_test, train_mask, val_mask, test_mask,labels = load_data(dataset)
# Train the GCN
QUICK_MODE = False
w_0, w_1, A_tilde, old_soft = get_trained_gcn(324, dataset, y_train, y_val, y_test, train_mask, val_mask,
                                                      test_mask)

A = adj.todense()
full_A_tilde = preprocess_adj(adj, True)

train_index = np.argwhere(train_mask).flatten()
val_index = np.argwhere(val_mask).flatten()
test_index = np.argwhere(test_mask).flatten()
features_sparse = preprocess_features(initial_features)
feature_matrix = features_sparse.todense()
number_nodes = feature_matrix.shape[0]
number_labels = labels.shape[1]
#list_new_posititons = range(number_nodes)
initial_gcn = old_soft(sparse_to_tuple(features_sparse))

full_pred_gcn = np.argmax(initial_gcn, axis=1)
ground_truth = np.argmax(labels, axis=1)
with open(result_path + intitial_gcn_neighbors_files, 'rb') as f:
    initial_neighbors_y = pk.load(f, encoding='latin1')


def percent_similar(list_nodes):
    percent_predicted_similar = []
    for i in list_nodes:
        pred_lab = full_pred_gcn[i]
        neighbors_labels = full_pred_gcn[np.argwhere(A[i])[:, 1]]
        similar_neighbors = np.where(neighbors_labels == pred_lab)[0].shape[0]
        num_neighbors = neighbors_labels.shape[0]
        percent_predicted_similar.append(similar_neighbors / num_neighbors)

    return percent_predicted_similar

def log_odds_ratio(v):
    p_max = v[np.argsort(v)[-1]]
    p_second_max = v[np.argsort(v)[-2]]
    return np.log((p_max * (1 - p_second_max)) / ((1 - p_max) * p_second_max))


log_odds_ratio_gcn = np.apply_along_axis(log_odds_ratio, 1, initial_gcn)

score = np.array(log_odds_ratio_gcn[test_index])

threshold = np.mean(score)
nodes_to_reclassify = test_index[np.argwhere(score < threshold)]
scores_reclassify = score[np.argwhere(score < threshold)]
print(nodes_to_reclassify.shape)

start_test_index = 1708
average_softmax_results = np.zeros((number_nodes, number_nodes, 7))  # Store raw softmax
print(average_softmax_results.shape)
print("Loading results...")
for softmax_files in onlyfiles:
    with open(result_path + softmax_files, 'rb') as f:
        partial_softmax_results = pk.load(f, encoding='latin1')
    average_softmax_results = average_softmax_results + partial_softmax_results
print("done")
print(average_softmax_results.shape)

new_pred_soft = deepcopy(full_pred_gcn)
new_pred_wei_soft = deepcopy(full_pred_gcn)
new_pred_log_neigh_wei = deepcopy(full_pred_gcn)
initial_avergae = np.mean(initial_gcn, axis=0)
neigh_average = np.mean(initial_neighbors_y, axis=0)
initial_neighbors_log = np.log(initial_neighbors_y)
initial_neigh_log_avg = np.mean(initial_neighbors_log, axis=0)
j = 0
for i in nodes_to_reclassify:
    all_output_for_node = average_softmax_results[i[0]]
    node_true_label = ground_truth[i]
    node_thinking_label = full_pred_gcn[i]

    y_bar_x = np.mean(all_output_for_node, axis=0)

    new_label = np.argmax(y_bar_x, axis=0)
    neighbors_labels = full_pred_gcn[np.argwhere(A[i])[:, 1]]
    similar_neighbors = np.where(neighbors_labels == new_label)[0].shape[0]
    num_neighbors = neighbors_labels.shape[0]

    if similar_neighbors / num_neighbors > scores_reclassify[j]:
        new_pred_soft[i] = new_label
        print(str(node_true_label) + " pred " + str(node_thinking_label) + " new : " + str(new_label))

    y_bar_x = y_bar_x - initial_avergae

    new_label = np.argmax(y_bar_x, axis=0)
    neighbors_labels = full_pred_gcn[np.argwhere(A[i])[:, 1]]
    similar_neighbors = np.where(neighbors_labels == new_label)[0].shape[0]
    num_neighbors = neighbors_labels.shape[0]

    if similar_neighbors / num_neighbors > scores_reclassify[j]:
        new_pred_wei_soft[i] = new_label
        print(str(node_true_label) + " pred " + str(node_thinking_label) + " new : " + str(new_label))

    log_all_output_for_node = np.log(all_output_for_node)
    b = log_all_output_for_node - initial_neighbors_log
    log_y_bar_weighted_neigh_x = np.mean(np.multiply(initial_neighbors_y, b), axis=0)

    new_label = np.argmax(log_y_bar_weighted_neigh_x, axis=0)
    neighbors_labels = full_pred_gcn[np.argwhere(A[i])[:, 1]]
    similar_neighbors = np.where(neighbors_labels == new_label)[0].shape[0]
    num_neighbors = neighbors_labels.shape[0]

    if similar_neighbors / num_neighbors > scores_reclassify[j]:
        new_pred_log_neigh_wei[i] = new_label
        print(str(node_true_label) + " pred " + str(node_thinking_label) + " new : " + str(new_label))

    #new_label = np.argmax(y_bar_x, axis=0)

    j += 1
print("ACC old pred : " + str(accuracy_score(ground_truth[test_index], full_pred_gcn[test_index])))

print("ACC soft  pred : " + str(accuracy_score(ground_truth[test_index], new_pred_soft[test_index])))

print("ACC corrected  pred : " + str(accuracy_score(ground_truth[test_index], new_pred_wei_soft[test_index])))

print("ACC log neigh  pred : " + str(accuracy_score(ground_truth[test_index], new_pred_log_neigh_wei[test_index])))
