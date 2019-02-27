import matplotlib.pyplot as plt


def plot_softmax_output(node_index, node_true_label, softmax_output_list, label_list):
    x = range(softmax_output_list.shape[1])
    print(x)
    for soft in softmax_output_list:
        plt.plot(x, soft, 'o')
    plt.show()