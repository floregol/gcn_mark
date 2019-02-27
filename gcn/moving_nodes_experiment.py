import random
import time
import tensorflow as tf

from utils import *
from models import GCN, MLP
import os
from scipy import sparse
from train import get_trained_gcn
from copy import copy, deepcopy
from plotting_results import plot_softmax_output
"""

 Moving the nodes around experiment

"""
# Train the GCN
sess, FLAGS, softmax = get_trained_gcn()

# Get features and labels
_, initial_features, _, _, _, train_mask, _, _, labels = load_data(FLAGS.dataset)
train_index = np.argwhere(train_mask).flatten()

features_sparse = preprocess_features(initial_features)
feature_matrix = features_sparse.todense()
number_nodes = feature_matrix.shape[0]
number_labels = labels.shape[1]

# Experiment parameters.
NUM_MOVED_NODES = 5
list_moved_node = random.sample(list(train_index), NUM_MOVED_NODES)
list_new_posititons = random.sample(range(number_nodes), 200)
#list_new_posititons = range(number_nodes)

for node_index in list_moved_node:  # TODO in parrallel copy features matrix

    node_features = deepcopy(feature_matrix[node_index])
    start_time = time.time()

    node_true_label = int(np.argwhere(labels[node_index]))
    print("Moving node " + str(node_index) + " with label " + str(node_true_label))

    # To store results
    softmax_output_list = np.zeros((len(list_new_posititons), number_labels))
    label_list = []
    i = 0

    for new_spot in list_new_posititons:

        replaced_node_label = int(np.argwhere(labels[new_spot]))
        label_list.append(replaced_node_label)  # to keep track of the label of the replaced node

        saved_features = deepcopy(
            feature_matrix[new_spot])  # save replaced node features to do everything in place (memory)

        feature_matrix[new_spot] = node_features  # move the node to the new position

        features = sparse_to_tuple(sparse.csr_matrix(feature_matrix))
        softmax_output_of_node = softmax(features, new_spot)  # get new softmax output at this position

        softmax_output_list[i] = softmax_output_of_node  # Store results
        i += 1
        print("put at " + str(replaced_node_label) + " = " + str(np.argmax(softmax_output_of_node)))

        feature_matrix[new_spot] = saved_features  # undo changes on the feature matrix

    # Create plot for this node
    plot_softmax_output(node_index, node_true_label, softmax_output_list, label_list)

    end_time = time.time()
    seconds_elapsed = end_time - start_time
