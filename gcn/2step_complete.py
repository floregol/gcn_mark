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
from sklearn.model_selection import StratifiedShuffleSplit
from helper import *
"""

 Moving the nodes around experiment

"""
NUM_CROSS_VAL = 4
trials = 2
# Train the GCN
SEED = 43
initial_num_labels = 20
THRESHOLD = 0.5
dataset = 'cora'
adj, initial_features, _, _, _, _, _, _, labels = load_data(dataset)
ground_truth = np.argmax(labels, axis=1)
A = adj.todense()
full_A_tilde = preprocess_adj(adj, True)
features_sparse = preprocess_features(initial_features)
feature_matrix = features_sparse.todense()
n = feature_matrix.shape[0]
number_labels = labels.shape[1]

list_new_posititons = random.sample(list(range(n)), 30)
#list_new_posititons = range(n)

test_split = StratifiedShuffleSplit(n_splits=NUM_CROSS_VAL, test_size=0.37, random_state=SEED)
test_split.get_n_splits(labels, labels)
seed_list = [1, 2, 3, 4]

for train_index, test_index in test_split.split(labels, labels):

    y_train, y_val, y_test, train_mask, val_mask, test_mask = get_split(n, train_index, test_index, labels,
                                                                        initial_num_labels)

    for trial in range(trials):
        seed = seed_list[trial]
        w_0, w_1, A_tilde, gcn_soft = get_trained_gcn(seed, dataset, y_train, y_val, y_test, train_mask, val_mask,
                                                      test_mask)

        # Get prediction by the GCN
        initial_gcn = gcn_soft(sparse_to_tuple(features_sparse))

        full_pred_gcn = np.argmax(initial_gcn, axis=1)
        new_pred_soft = deepcopy(full_pred_gcn)
        new_pred_wei_soft = deepcopy(full_pred_gcn)
        new_pred_log_neigh_wei = deepcopy(full_pred_gcn)

        print("ACC old pred : " + str(accuracy_score(ground_truth[test_index], full_pred_gcn[test_index])))

        initial_avergae = np.mean(initial_gcn, axis=0)

        # initial_neighbors_y =
        # neigh_average = np.mean(initial_neighbors_y, axis=0)
        # initial_neighbors_log = np.log(initial_neighbors_y)
        # initial_neigh_log_avg = np.mean(initial_neighbors_log, axis=0)
        def log_odds_ratio(v):
            p_max = v[np.argsort(v)[-1]]
            p_second_max = v[np.argsort(v)[-2]]
            return np.log((p_max * (1 - p_second_max)) / ((1 - p_max) * p_second_max))


        log_odds_ratio_gcn = np.apply_along_axis(log_odds_ratio, 1, initial_gcn)

        score = np.array(log_odds_ratio_gcn[test_index])

        threshold = np.mean(score)
        score = np.array(score_percent_similar(test_index, full_pred_gcn, A))
        nodes_to_reclassify = test_index[np.argwhere(score < THRESHOLD)]
        scores_reclassify = score[np.argwhere(score < THRESHOLD)]
        print(nodes_to_reclassify.shape)
        j = 0
        for node_index in nodes_to_reclassify:  # TODO in parrallel copy features matrix

            node_features = deepcopy(feature_matrix[node_index])
            start_time = time.time()

            node_true_label = ground_truth[node_index]
            node_thinking_label = full_pred_gcn[node_index]
            #To store results
            softmax_output_list = np.zeros((len(list_new_posititons), number_labels))

            label_list = []
            i = 0

            for new_spot in list_new_posititons:

                replaced_node_label = int(np.argwhere(labels[new_spot]))
                label_list.append(replaced_node_label)  # to keep track of the label of the replaced node

                saved_features = deepcopy(
                    feature_matrix[new_spot])  # save replaced node features to do everything in place (memory)

                feature_matrix[new_spot] = node_features  # move the node to the new position

                softmax_output_of_node = fast_localized_softmax(feature_matrix, new_spot, full_A_tilde, w_0,
                                                                w_1)  # get new softmax output at this position

                softmax_output_list[i] = softmax_output_of_node  # Store results
                i += 1
                # print("put at " + str(replaced_node_label) + " = " + str(np.argmax(softmax_output_of_node)))

                feature_matrix[new_spot] = saved_features  # undo changes on the feature matrix

            all_output_for_node = softmax_output_list
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

            # log_all_output_for_node = np.log(all_output_for_node)
            # b = log_all_output_for_node - initial_neighbors_log
            # log_y_bar_weighted_neigh_x = np.mean(np.multiply(initial_neighbors_y, b), axis=0)

            # new_label = np.argmax(log_y_bar_weighted_neigh_x, axis=0)
            # neighbors_labels = full_pred_gcn[np.argwhere(A[i])[:, 1]]
            # similar_neighbors = np.where(neighbors_labels == new_label)[0].shape[0]
            # num_neighbors = neighbors_labels.shape[0]

            # if similar_neighbors / num_neighbors > scores_reclassify[j]:
            #     new_pred_log_neigh_wei[i] = new_label
            #     print(str(node_true_label) + " pred " + str(node_thinking_label) + " new : " + str(new_label))

            #new_label = np.argmax(y_bar_x, axis=0)

            j += 1

        print("ACC old pred : " + str(accuracy_score(ground_truth[test_index], full_pred_gcn[test_index])))

        print("ACC soft  pred : " + str(accuracy_score(ground_truth[test_index], new_pred_soft[test_index])))

        print("ACC corrected  pred : " + str(accuracy_score(ground_truth[test_index], new_pred_wei_soft[test_index])))

        # print("ACC log neigh  pred : " +
        #       str(accuracy_score(ground_truth[test_index], new_pred_log_neigh_wei[test_index])))
