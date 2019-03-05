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
"""

 Moving the nodes around experiment

"""

# Train the GCN
QUICK_MODE = False
sess, FLAGS, softmax = get_trained_gcn(QUICK_MODE)
result_folder = "results/classification"
# Get features and labels
adj, initial_features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = load_data(FLAGS.dataset)

train_index = np.argwhere(train_mask).flatten()
val_index = np.argwhere(val_mask).flatten()
test_index = np.argwhere(test_mask).flatten()
features_sparse = preprocess_features(initial_features)
feature_matrix = features_sparse.todense()
number_nodes = feature_matrix.shape[0]
number_labels = labels.shape[1]
# nodes_to_classify = test_index
# list_new_posititons = range(number_nodes)
list_new_posititons = random.sample(list(range(number_nodes)), 10)
nodes_to_classify = random.sample(list(test_index), 3)
j = 0

features = sparse_to_tuple(sparse.csr_matrix(feature_matrix))
y_bar = np.mean(softmax(features, list_new_posititons), axis=0)

average_softmax_results = np.zeros((number_nodes, number_labels))

for node_index in nodes_to_classify:  # TODO in parrallel copy features matrix

    node_features = deepcopy(feature_matrix[node_index])
    start_time = time.time()

    node_true_label = int(np.argwhere(labels[node_index]))
    print("lABEL : " + str(node_true_label))
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
        # print("put at " + str(replaced_node_label) + " = " + str(np.argmax(softmax_output_of_node)))

        feature_matrix[new_spot] = saved_features  # undo changes on the feature matrix

    average_softmax_node = np.mean(softmax_output_list, axis=0)
    average_softmax_results[node_index] = average_softmax_node

# Store data
pk.dump(average_softmax_results, open(os.path.join(result_folder, "softmax.pk"), 'wb'))