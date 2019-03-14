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
"""

 Moving the nodes around experiment

"""

# Train the GCN
QUICK_MODE = False
w_0, w_1, A_tilde, FLAGS, old_softmax = get_trained_gcn(QUICK_MODE)

A_index = A_tilde[0][0]
A_values = A_tilde[0][1]
A_shape = A_tilde[0][2]
result_folder = "results/classification"
# Get features and labels
adj, initial_features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = load_data(FLAGS.dataset)

full_A_tilde = preprocess_adj(adj, True)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1)


def fast_localized_softmax(features, new_spot):
    neighbors_index = np.argwhere(full_A_tilde[new_spot, :])[:, 1]
    A_neig = full_A_tilde[neighbors_index, :]
    H_out = np.matmul(np.matmul(A_neig, features), w_0)
    relu_out = np.maximum(H_out, 0, H_out)
    A_i_neigh = full_A_tilde[new_spot, neighbors_index]
    H2_out = np.matmul(np.matmul(A_i_neigh, relu_out), w_1)
    return softmax(H2_out)


train_index = np.argwhere(train_mask).flatten()
val_index = np.argwhere(val_mask).flatten()
test_index = np.argwhere(test_mask).flatten()
features_sparse = preprocess_features(initial_features)
feature_matrix = features_sparse.todense()
number_nodes = feature_matrix.shape[0]
number_labels = labels.shape[1]

nodes_to_classify = test_index[int(sys.argv[1]):int(sys.argv[2])]
list_new_posititons = range(number_nodes)
#list_new_posititons = random.sample(list(range(number_nodes)), 30)
# nodes_to_classify = random.sample(list(test_index), 3)
j = 0

features = sparse_to_tuple(sparse.csr_matrix(feature_matrix))



softmax_results = np.zeros((number_nodes, len(list_new_posititons), number_labels))

for node_index in nodes_to_classify:  # TODO in parrallel copy features matrix

    node_features = deepcopy(feature_matrix[node_index])
    start_time = time.time()

    node_true_label = int(np.argwhere(labels[node_index]))
    print(str(j) + "/" + str(len(nodes_to_classify)))
    # To store results
    softmax_output_list = np.zeros((len(list_new_posititons), number_labels))
    #classes_count = np.zeros((1, number_labels))
    label_list = []
    i = 0

    for new_spot in list_new_posititons:

        replaced_node_label = int(np.argwhere(labels[new_spot]))
        label_list.append(replaced_node_label)  # to keep track of the label of the replaced node

        saved_features = deepcopy(
            feature_matrix[new_spot])  # save replaced node features to do everything in place (memory)

        feature_matrix[new_spot] = node_features  # move the node to the new position

        #features = sparse_to_tuple(sparse.csr_matrix(feature_matrix))
        softmax_output_of_node = fast_localized_softmax(feature_matrix,
                                                        new_spot)  # get new softmax output at this position
        # softmax_output_of_node = old_softmax(features,new_spot)

        # classes_count[0, np.argmax(softmax_output_of_node)] = classes_count[0, np.argmax(softmax_output_of_node)] + 1
        softmax_output_list[i] = softmax_output_of_node  # Store results
        i += 1
        # print("put at " + str(replaced_node_label) + " = " + str(np.argmax(softmax_output_of_node)))

        feature_matrix[new_spot] = saved_features  # undo changes on the feature matrix
    j += 1
    softmax_results[node_index] = softmax_output_list
    #classes_results[node_index] = classes_count

# Store data
pk.dump(softmax_results, open(os.path.join(result_folder, "full" + sys.argv[1] + ".pk"), 'wb'))
