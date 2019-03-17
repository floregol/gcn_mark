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
# Load result files
onlyfiles = [f for f in listdir(result_path) if isfile(join(result_path, f))]
onlyfiles.remove(intitial_gcn_neighbors_files)
onlyfiles.remove(intitial_gcn_files)

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
    y_bar_x_train = np.mean(all_output_for_node[train_position_considered], axis=0)
    y_bar_x_weighted_train = y_bar_x_train - train_y_average

    a = log_all_output_for_node - initial_log
    b = log_all_output_for_node - initial_neighbors_log

    log_y_bar_x = np.mean(a, axis=0)
    log_y_bar_x_train = np.mean(a[train_position_considered], axis=0)
    log_y_bar_neigh_x = np.mean(b, axis=0)

    log_y_bar_weighted_x = np.mean(np.multiply(initial_y, a), axis=0)
    log_y_bar_weighted_x_train = np.mean(
        np.multiply(initial_y[train_position_considered], a[train_position_considered]), axis=0)
    log_y_bar_weighted_neigh_x = np.mean(np.multiply(initial_neighbors_y, b), axis=0)

    classifier_avg_softmax.append(np.argmax(y_bar_x))
    classifier_avg_wei_softmax.append(np.argmax(y_bar_x_weighted))
    classifier_log.append(np.argmax(log_y_bar_x))
    classifier_log_neigh.append(np.argmax(log_y_bar_neigh_x))
    classifier_wei_log.append(np.argmax(log_y_bar_weighted_x))
    classifier_wei_log_neigh.append(np.argmax(log_y_bar_weighted_neigh_x))

    classifier_train_avg.append(np.argmax(y_bar_x_train))
    classifier_train_avg_wei.append(np.argmax(y_bar_x_weighted_train))

    classifier_train_log.append(np.argmax(log_y_bar_x_train))
    classifier_train_wei_log.append(np.argmax(log_y_bar_weighted_x_train))

# print("Average Softmax ")
# print(accuracy_score(classifier_avg_softmax, label_ground_truth))
# print(f1_score(classifier_avg_softmax, label_ground_truth, average="macro"))
# print("Average Weighted Softmax ")
# print(accuracy_score(classifier_avg_wei_softmax, label_ground_truth))
# print(f1_score(classifier_avg_wei_softmax, label_ground_truth, average="macro"))
# print("Average Log ")
# print(accuracy_score(classifier_log, label_ground_truth))
# print(f1_score(classifier_log, label_ground_truth, average="macro"))
# print("Average Log Neighbors ")
# print(accuracy_score(classifier_log_neigh, label_ground_truth))
# print(f1_score(classifier_log_neigh, label_ground_truth, average="macro"))
# print("Average Log Weighted ")
# print(accuracy_score(classifier_wei_log, label_ground_truth))
# print(f1_score(classifier_wei_log, label_ground_truth, average="macro"))
# print("Average Log Weighted Neighbors ")
# print(accuracy_score(classifier_wei_log_neigh, label_ground_truth))
# print(f1_score(classifier_wei_log_neigh, label_ground_truth, average="macro"))
# print("Train Average Softmax ")
# print(accuracy_score(classifier_train_avg, label_ground_truth))
# print(f1_score(classifier_train_avg, label_ground_truth, average="macro"))
# print("Train Average Weighted Softmax ")
# print(accuracy_score(classifier_train_avg_wei, label_ground_truth))
# print(f1_score(classifier_train_avg_wei, label_ground_truth, average="macro"))
# print("Train Average Log ")
# print(accuracy_score(classifier_train_log, label_ground_truth))
# print(f1_score(classifier_train_log, label_ground_truth, average="macro"))
# print("Train Average Log Weighted ")
# print(accuracy_score(classifier_train_wei_log, label_ground_truth))
# print(f1_score(classifier_train_wei_log, label_ground_truth, average="macro"))
# print("Train Top softmax ")
# print(accuracy_score(classifier_top_soft, label_ground_truth))
# print(f1_score(classifier_top_soft, label_ground_truth, average="macro"))

gcn_classifier = np.argmax(initial_y[1708:2708], axis=1)
gcn_good = np.where(gcn_classifier == label_ground_truth)[0]
gcn_not = np.where(gcn_classifier != label_ground_truth)[0]

avg_softmax_wei_good = np.where(classifier_avg_wei_softmax == label_ground_truth)[0]
avg_softmax_wei_not = np.where(classifier_avg_wei_softmax != label_ground_truth)[0]

still_good = []
now_good = []
now_bad = []
still_bad = []

for i in range(1000):
    real_index = i + 1708
    if i in gcn_good:
        if i in avg_softmax_wei_good:
            still_good.append(real_index)
        else:
            now_bad.append(real_index)
    else:
        if i in avg_softmax_wei_good:
            now_good.append(real_index)
        else:
            still_bad.append(real_index)
print()
print("--------------------")
print()
print("Num still good " + str(len(still_good)))
print("Num still bad " + str(len(still_bad)))
print("Num now good " + str(len(now_good)))
print("Num now bad " + str(len(now_bad)))
print()
print("--------------------")
print()

print("Avg Entropy still good " + str(np.mean(entropy_gcn[still_good])) + "+/-" + str(np.std(entropy_gcn[still_good])))
print("Avg Entropy still bad " + str(np.mean(entropy_gcn[still_bad])) + "+/-" + str(np.std(entropy_gcn[still_bad])))
print("Avg Entropy now good " + str(np.mean(entropy_gcn[now_good])) + "+/-" + str(np.std(entropy_gcn[now_good])))
print("Avg Entropy now bad " + str(np.mean(entropy_gcn[now_bad])) + "+/-" + str(np.std(entropy_gcn[now_bad])))
print()
print("--------------------")
print()
print("avergae degree still good " + str(np.sum(A[still_good]) / len(still_good)))
print("avergae degree still bad " + str(np.sum(A[still_bad]) / len(still_bad)))
print("avergae degree now good " + str(np.sum(A[now_good]) / len(now_good)))
print("avergae degree now bad " + str(np.sum(A[now_bad]) / len(now_bad)))
print()
print("--------------------")
print()

ground_truth = np.argmax(labels, axis=1)


def percent_similar(list_nodes):
    percent_similar = []
    for i in list_nodes:
        real_lab = ground_truth[i]
        neighbors_labels = ground_truth[np.argwhere(A[i])[:, 1]]
        similar_neighbors = np.where(neighbors_labels == real_lab)[0].shape[0]
        num_neighbors = neighbors_labels.shape[0]
        percent_similar.append(similar_neighbors / num_neighbors)
    print(np.mean(percent_similar))
    percent_similar = []


percent_similar(still_good)
percent_similar(still_bad)
percent_similar(now_good)
percent_similar(now_bad)
