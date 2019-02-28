import matplotlib.pyplot as plt
import pickle as pk
import numpy as np
import os
from os import listdir
from os.path import isfile, join


def plot_softmax_output(node_index, node_true_label, softmax_output_list, label_list):

    plt.figure()

    x = range(softmax_output_list.shape[1])
    for l in range(7):
        data = softmax_output_list[:, l]
        plt.boxplot(data, positions=[l], widths=0.6)

    plt.axis([-0.5, 6.5, 0, 1])
    ticks = [0, 1, 2, 3, 4, 5, 6]
    plt.xticks(range(len(ticks)), ticks)
    plt.title("Softmax Histogram for Node " + str(node_index) + " Label : " + str(node_true_label))
    plt.savefig("plots/node_" + str(node_index) + "_label" + str(node_true_label) + ".png")


mypath = 'results/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for node_files in onlyfiles:
    with open("results/" + node_files, 'rb') as f:
        stats_dict = pk.load(f, encoding='latin1')

    node_index = stats_dict["node"]
    node_true_label = stats_dict["node_label"]
    softmax_output_list = stats_dict["softmax"]
    label_list = stats_dict["label_list"]
    plot_softmax_output(node_index, node_true_label, softmax_output_list, label_list)
