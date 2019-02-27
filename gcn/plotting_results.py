import matplotlib.pyplot as plt
import pickle as pk
#plot_softmax_output(node_index, node_true_label, softmax_output_list, label_list)

#with open("results/node_83.pk", 'rb') as f:
#    stats_dict = pk.load(f, encoding='latin1')

#print(stats_dict)
# def plot_softmax_output(node_index, node_true_label, softmax_output_list, label_list):
#     x = range(softmax_output_list.shape[1])
#     print(x)
#     for soft in softmax_output_list:
#         plt.plot(x, soft,'o',alpha = 0.5, color='k')
        
#     # fake up some data
#     spread = np.random.rand(50) * 100
#     center = np.ones(25) * 50
#     flier_high = np.random.rand(10) * 100 + 100
#     flier_low = np.random.rand(10) * -100
#     data = np.concatenate((spread, center, flier_high, flier_low))
    
    
#     plt.boxplot(data)
#     plt.axis([-0.5,6.5, 0, 1])
#     plt.show()
#     #plt.savefig("node_"+str(node_index)+"_label"+ str(node_true_label)+".png")
    
