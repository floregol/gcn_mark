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

nodes_to_classify = test_index
list_new_posititons = range(number_nodes)
# list_new_posititons = random.sample(list(range(number_nodes)), 10)
# nodes_to_classify = random.sample(list(test_index), 3)
j = 0

features = sparse_to_tuple(sparse.csr_matrix(feature_matrix))
softmax_output_test = softmax(features, test_index)
print(softmax_output_test)
print(softmax_output_test.shape)

label_predictions = np.argmax(softmax_output_test, axis=1)
label_ground_truth = np.argmax(labels[test_index, :], axis=1)

print(accuracy_score(label_predictions, label_ground_truth))

args_good = np.where(label_predictions == label_ground_truth)
args_not = np.where(label_predictions != label_ground_truth)
# print(args_good[0])
# print(args_not[0])
correct_softmax = softmax_output_test[args_good, :][0]

not_correct_softmax = softmax_output_test[args_not, :][0]

entropy_list = np.apply_along_axis(entropy, 1, correct_softmax)
entropy_list_bad = np.apply_along_axis(entropy, 1, not_correct_softmax)
max_entropy = math.log(7)
print(np.mean(entropy_list) / max_entropy)
print(np.mean(entropy_list_bad) / max_entropy)

y_bar = np.mean(softmax(features, list_new_posititons), axis=0)
print(y_bar)
average_softmax_results = np.zeros((number_nodes, number_labels))

for node_index in nodes_to_classify:  # TODO in parrallel copy features matrix

    node_features = deepcopy(feature_matrix[node_index])
    start_time = time.time()

    node_true_label = int(np.argwhere(labels[node_index]))
    print(str(j) + "/" + str(len(nodes_to_classify)))
    # To store results
    #softmax_output_list = np.zeros((len(list_new_posititons), number_labels))
    classes_count = np.zeros((1, number_labels))
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
        classes_count[0, np.argmax(softmax_output_of_node)] = classes_count[0, np.argmax(softmax_output_of_node)] + 1
        #  softmax_output_list[i] = softmax_output_of_node  # Store results
        i += 1
        # print("put at " + str(replaced_node_label) + " = " + str(np.argmax(softmax_output_of_node)))

        feature_matrix[new_spot] = saved_features  # undo changes on the feature matrix
    j += 1
    # average_softmax_node = np.mean(softmax_output_list, axis=0)
    # average_softmax_results[node_index] = average_softmax_node
    classes_results[node_index] = classes_count

# Store data
pk.dump(classes_results, open(os.path.join(result_folder, "class" + sys.argv[1] + ".pk"), 'wb'))
