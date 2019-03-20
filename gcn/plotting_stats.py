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

adj, initial_features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = load_data('cora')
A = adj.todense()

print("Loading initial position output...")
with open(result_path + intitial_gcn_files, 'rb') as f:
    initial_y = pk.load(f, encoding='latin1')

with open(result_path + intitial_gcn_neighbors_files, 'rb') as f:
    initial_neighbors_y = pk.load(f, encoding='latin1')

start_test_index = 1708
num_nodes = 2708

label_ground_truth = np.argmax(y_test[start_test_index:num_nodes], axis=1)


def log_odds_ratio(v):
    p_max = v[np.argsort(v)[-1]]
    p_second_max = v[np.argsort(v)[-2]]
    return np.log((p_max * (1 - p_second_max)) / ((1 - p_max) * p_second_max))


def KL_normalized(v):
    return 0


entropy_gcn = np.apply_along_axis(entropy, 1, initial_y)
log_odds_ratio_gcn = np.apply_along_axis(log_odds_ratio, 1, initial_y)
KL_normalized = np.apply_along_axis(KL_normalized, 1, initial_y)

position_considered = list(range(0, 2708))
train_position_considered = list(range(0, 1708))

print("Loading Classification...")
with open(result_path + classifier_avg_softmax_file, 'rb') as f:
    classifier_avg_softmax = pk.load(f, encoding='latin1')
with open(result_path + classifier_avg_wei_softmax_file, 'rb') as f:
    classifier_avg_wei_softmax = pk.load(f, encoding='latin1')


def print_acc_f1(name, classifier_output):
    print(name)
    print(accuracy_score(classifier_output, label_ground_truth))
    print(f1_score(classifier_output, label_ground_truth, average="macro"))


print_acc_f1("Average Softmax ", classifier_avg_softmax)
print_acc_f1("Average Weighted Softmax ", classifier_avg_wei_softmax)
# print_acc_f1("Average Log ", classifier_avg_wei_softmax_nei)
# print_acc_f1("Average Weighted Softmax Neighbors", classifier_avg_wei_softmax_nei)
# print_acc_f1("Average Log ", classifier_log)
# print_acc_f1("Average Log Neighbors ", classifier_log_neigh)
# print_acc_f1("Average Log Weighted ", classifier_wei_log)
# print_acc_f1("Average Log Weighted Neighbors ", classifier_wei_log_neigh)
# print_acc_f1("Train Average Softmax ", classifier_train_avg)
# print_acc_f1("Train Average Weighted Softmax ", classifier_train_avg_wei)
# print_acc_f1("Train Average Log ", classifier_train_log)
# print_acc_f1("Train Average Log Weighted ", classifier_train_wei_log)
# print_acc_f1("Train Top softmax ", classifier_top_soft)

gcn_classifier = np.argmax(initial_y[1708:2708], axis=1)
gcn_good = np.where(gcn_classifier == label_ground_truth)[0]
gcn_not = np.where(gcn_classifier != label_ground_truth)[0]
# classifier_avg_wei_softmax
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


def histo(name_metric, list_metric):
    plt.hist(list_metric[still_good], alpha=0.5, color='b', bins=60)
    plt.hist(list_metric[still_bad], alpha=0.5, color='k', bins=60)
    plt.xlabel(name_metric)
    plt.show()
    # plt.savefig(name_metric + "_still.png", dpi=300)
    plt.close()
    plt.hist(list_metric[now_bad], alpha=0.5, color='r', bins=60)
    plt.hist(list_metric[now_good], alpha=0.5, color='g', bins=60)
    plt.xlabel(name_metric)
    plt.show()
    #plt.savefig(name_metric + "_changed.png", dpi=300)
    plt.close()


ground_truth = np.argmax(labels, axis=1)

full_pred_gcn = np.argmax(initial_y, axis=1)


def percent_similar(list_nodes):
    percent_similar = []
    percent_predicted_similar = []
    for i in list_nodes:
        real_lab = ground_truth[i]
        pred_lab = full_pred_gcn[i]

        neighbors_labels = ground_truth[np.argwhere(A[i])[:, 1]]
        similar_neighbors = np.where(neighbors_labels == real_lab)[0].shape[0]
        num_neighbors = neighbors_labels.shape[0]
        percent_similar.append(similar_neighbors / num_neighbors)

        neighbors_labels = full_pred_gcn[np.argwhere(A[i])[:, 1]]
        similar_neighbors = np.where(neighbors_labels == pred_lab)[0].shape[0]
        num_neighbors = neighbors_labels.shape[0]
        percent_predicted_similar.append(similar_neighbors / num_neighbors)

    return percent_predicted_similar


def KL_normalized(list_nodes):
    KL = []
    for i in list_nodes:
        p = initial_y[i]
        neighbors_p = initial_y[np.argwhere(A[i])[:, 1]]
        KL_neighbors = []
        for n in neighbors_p:
            KL_neighbors.append(0.5 * entropy(p, n) + 0.5 * entropy(n, p))
        num_neighbors = neighbors_p.shape[0]
        KL.append(np.mean(KL_neighbors) / num_neighbors)
    return KL


def KL_top_normalized(list_nodes):
    KL = []
    for i in list_nodes:
        p = initial_y[i]
        top_2_index = np.argsort(p)[-2:]
        low = np.full(p.shape, 0.0001)
        low[top_2_index] = p[top_2_index]
        neighbors_p = initial_y[np.argwhere(A[i])[:, 1]]
        KL_neighbors = []
        for n in neighbors_p:
            top_2_index = np.argsort(n)[-2:]
            low_neigh = np.full(p.shape, 0.0001)
            low_neigh[top_2_index] = n[top_2_index]
            KL_neighbors.append(0.5 * entropy(low, low_neigh) + 0.5 * entropy(low_neigh, low))
        num_neighbors = neighbors_p.shape[0]
        KL.append(np.mean(KL_neighbors) / num_neighbors)
    return KL


#percent_stuff = np.array(percent_similar(range(0, 2708)))
# KL_normalized = np.array(KL_normalized(range(0, 2708)))
# histo("KL_normalized", KL_normalized)

# percent_similar_gcn_still_good = percent_similar(still_good)
# percent_similar_gcn_still_bad = percent_similar(still_bad)
# percent_similar_gcn_now_good = percent_similar(now_good)
# percent_similar_gcn_now_bad = percent_similar(now_bad)

# print()
# print("--------------------")
# print()
# print("Num still good " + str(len(still_good)))
# print("Num still bad " + str(len(still_bad)))
# print("Num now good " + str(len(now_good)))
# print("Num now bad " + str(len(now_bad)))
# print()
# print("--------------------")
# print()

# print("Avg Entropy still good " + str(np.mean(entropy_gcn[still_good])) + "+/-" + str(np.std(entropy_gcn[still_good])))
# print("Avg Entropy still bad " + str(np.mean(entropy_gcn[still_bad])) + "+/-" + str(np.std(entropy_gcn[still_bad])))
# print("Avg Entropy now good " + str(np.mean(entropy_gcn[now_good])) + "+/-" + str(np.std(entropy_gcn[now_good])))
# print("Avg Entropy now bad " + str(np.mean(entropy_gcn[now_bad])) + "+/-" + str(np.std(entropy_gcn[now_bad])))

# plt.hist(entropy_gcn[still_good], alpha=0.5, color='b', bins='auto')
# plt.hist(entropy_gcn[still_bad], alpha=0.5, color='k', bins='auto')
# plt.xlabel("Entropy")
# plt.savefig("entropy_still.png", dpi=300)
# plt.close()
# plt.hist(entropy_gcn[now_bad], alpha=0.5, color='r', bins='auto')
# plt.hist(entropy_gcn[now_good], alpha=0.5, color='g', bins='auto')
# plt.xlabel("Entropy")
# plt.savefig("entropy_changed.png", dpi=300)
# plt.close()
# print()
# print("--------------------")
# print()
# print("avergae degree still good " + str(np.sum(A[still_good]) / len(still_good)))
# print("avergae degree still bad " + str(np.sum(A[still_bad]) / len(still_bad)))
# print("avergae degree now good " + str(np.sum(A[now_good]) / len(now_good)))
# print("avergae degree now bad " + str(np.sum(A[now_bad]) / len(now_bad)))
# print()
# print("--------------------")
# print()

# plt.hist(np.sum(A[still_good], axis=1), alpha=0.5, color='b', bins='auto')
# plt.hist(np.sum(A[still_bad], axis=1), alpha=0.5, color='k', bins='auto')
# plt.xlabel("Degree")
# plt.ylabel("Num. Nodes")
# plt.savefig("degree_still.png", dpi=300)
# plt.close()
# plt.hist(np.sum(A[now_bad], axis=1), alpha=0.5, color='r', bins='auto')
# plt.hist(np.sum(A[now_good], axis=1), alpha=0.5, color='g', bins='auto')
# plt.xlabel("Degree")
# plt.ylabel("Num. Nodes")
# plt.savefig("degree_changed.png", dpi=300)
# plt.close()

# plt.hist(percent_similar_gcn_still_good, alpha=0.5, color='b', bins='auto')
# plt.hist(percent_similar_gcn_still_bad, alpha=0.5, color='k', bins='auto')
# plt.xlabel("Percent Predicted in Neighbors Agreement")
# plt.show()
# plt.close()
# plt.hist(percent_similar_gcn_now_good, alpha=0.5, color='g', bins='auto')
# plt.hist(percent_similar_gcn_now_bad, alpha=0.5, color='r', bins='auto')
# plt.xlabel("Percent Predicted in Neigbors Agreement")
# plt.show()
# plt.close()