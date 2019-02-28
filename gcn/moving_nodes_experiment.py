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
"""

 Moving the nodes around experiment

"""
# Train the GCN
sess, FLAGS, softmax = get_trained_gcn()
result_folder = "results"
# Get features and labels
adj, initial_features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = load_data(FLAGS.dataset)

features_sparse = preprocess_features(initial_features)
feature_matrix = features_sparse.todense()
number_nodes = feature_matrix.shape[0]
number_labels = labels.shape[1]

# Experiment parameters.
NUM_MOVED_NODES = 50

SET = "val"  # val, train_labeled, train_unlabeled
if SET == "train_labeled":
    set_index = np.argwhere(train_mask).flatten()
    result_folder += "/train_labeled"
elif SET == "train_unlabeled":
    pass
elif SET == "val":
    set_index = np.argwhere(val_mask).flatten()
    result_folder += "/val"
elif SET == "test":
    set_index = np.argwhere(test_mask).flatten()
    result_folder += "/test"

list_moved_node = random.sample(list(set_index), NUM_MOVED_NODES)
list_new_posititons = random.sample(range(number_nodes), 10)
#list_new_posititons = range(number_nodes)
j = 0
for node_index in list_moved_node:  # TODO in parrallel copy features matrix
    print(str(j) + "/" + str(len((list_moved_node))))
    j += 1
    node_features = deepcopy(feature_matrix[node_index])
    start_time = time.time()

    node_true_label = int(np.argwhere(labels[node_index]))
    #print("Moving node " + str(node_index) + " with label " + str(node_true_label))

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

    # Store data to create plots
    dict_output = {
        "node": node_index,
        "node_label": node_true_label,
        "softmax": softmax_output_list,
        "label_list": label_list
    }
    pk.dump(dict_output, open(os.path.join(result_folder, "node_" + str(node_index) + ".pk"), 'wb'))

    end_time = time.time()
    seconds_elapsed = end_time - start_time
