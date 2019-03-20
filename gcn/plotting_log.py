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

adj, initial_features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = load_data('cora')
A = adj.todense()

start_test_index = 1708
num_nodes = 2708

label_ground_truth = np.argmax(y_test[start_test_index:num_nodes], axis=1)

average_softmax_results = np.zeros((1000, num_nodes, 7))  # Store raw softmax
print("Loading results...")
for softmax_files in onlyfiles:
    with open(result_path + softmax_files, 'rb') as f:
        partial_softmax_results = pk.load(f, encoding='latin1')
    average_softmax_results = average_softmax_results + partial_softmax_results[start_test_index:num_nodes, :, :]


print("Loading initial position output...")
with open(result_path + intitial_gcn_files, 'rb') as f:
    initial_y = pk.load(f, encoding='latin1')

with open(result_path + intitial_gcn_neighbors_files, 'rb') as f:
    initial_neighbors_y = pk.load(f, encoding='latin1')

entropy_gcn = np.apply_along_axis(entropy, 1, initial_y)

position_considered = list(range(0, 2708))
train_position_considered = list(range(0, 1708))

initial_log = np.log(initial_y)
initial_neighbors_log = np.log(initial_neighbors_y)

classifier_avg_softmax = []
classifier_avg_wei_softmax = []
classifier_avg_wei_softmax_nei = []
classifier_log = []
classifier_log_neigh = []
classifier_wei_log = []
classifier_wei_log_neigh = []
classifier_train_avg = []
classifier_train_avg_wei = []
classifier_train_log = []
classifier_train_wei_log = []
classifier_top_soft = []
initial_y_average = np.mean(initial_y[position_considered], axis=0)
initial_neighbors_y_average = np.mean(initial_neighbors_y[position_considered], axis=0)
train_y_average = np.mean(initial_y[train_position_considered], axis=0)
initial_log_avg = np.mean(initial_log[position_considered], axis=0)
initial_neigh_log_avg = np.mean(initial_neighbors_log[position_considered], axis=0)

for node_index in range(1000):
    all_output_for_node = average_softmax_results[node_index]
    # top_softmax = []
    # only_top = []
    # for t in all_output_for_node:
    #     r = np.argmax(t)
    #     if t[r] >= 0.8:
    #         top_softmax.append(t)

    # top_soft = np.mean(top_softmax, axis=0)
    # classifier_top_soft.append(np.argmax(top_soft))

    log_all_output_for_node = np.log(all_output_for_node)

    y_bar_x = np.mean(all_output_for_node, axis=0)
    y_bar_x_weighted = y_bar_x - initial_y_average
    y_bar_x_weighted_nei = y_bar_x - initial_neighbors_y_average
    y_bar_x_train = np.mean(all_output_for_node[train_position_considered], axis=0)
    y_bar_x_weighted_train = y_bar_x_train - train_y_average

    a = log_all_output_for_node - initial_log
    b = log_all_output_for_node - initial_neighbors_log

    log_y_bar_x = np.mean(a, axis=0)
    log_y_bar_x_train = np.mean(a[train_position_considered], axis=0)
    log_y_bar_neigh_x = np.mean(b, axis=0)

    log_y_bar_weighted_x = np.mean(np.multiply(initial_y, a), axis=0)
    log_y_bar_weighted_x_train = np.mean(_considered], a[train_position_considered]), axis=0)
    log_y_bar_weighted_neigh_x = np.mean(np.multiply(initial_neighbors_y, b), axis=0)

    classifier_avg_softmax.append(np.argmax(y_bar_x))
    classifier_avg_wei_softmax.append(np.argmax(y_bar_x_weighted))
    classifier_avg_wei_softmax_nei.append(np.argmax(y_bar_x_weighted_nei))
    classifier_log.append(np.argmax(log_y_bar_x))
    classifier_log_neigh.append(np.argmax(log_y_bar_neigh_x))
    classifier_wei_log.append(np.argmax(log_y_bar_weighted_x))
    classifier_wei_log_neigh.append(np.argmax(log_y_bar_weighted_neigh_x))

    classifier_train_avg.append(np.argmax(y_bar_x_train))
    classifier_train_avg_wei.append(np.argmax(y_bar_x_weighted_train))

    classifier_train_log.append(np.argmax(log_y_bar_x_train))
    classifier_train_wei_log.append(np.argmax(log_y_bar_weighted_x_train))

print(accuracy_score(classifier_wei_log_neigh, label_ground_truth))
print(f1_score(classifier_wei_log_neigh, label_ground_truth, average="macro"))