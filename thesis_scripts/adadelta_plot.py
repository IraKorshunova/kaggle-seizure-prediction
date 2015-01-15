import json
from test_labels_loader.config_name_creator import create_cnn_model_name
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame


def plot_learning_curves(file_name1, file_name2):
    with open(file_name1, 'rb') as f:
        d1 = np.loadtxt(f)

    with open(file_name2, 'rb') as f:
        d2 = np.loadtxt(f)

    ax = plt.gca()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)
    ax.set_xticks([2500,5000,7500, 10000, 12500, 15000, 17500])
    ax.set_xticklabels([25,50,75,100,125,150,175])
    plt.xlim([0, 17500])
    plt.ylim([0, 0.55])
    plt.xlabel('Iterations, $10^3$', fontsize=20)
    plt.ylabel('Negative log-likelihood', fontsize=20)
    plt.plot(d1, color='grey', label='ADADELTA 1.0')
    plt.plot(d2, linewidth=1.3, color='black', label='ADADELTA 0.01')
    plt.legend(loc='upper right')
    plt.show()


def plot_learning_curve(file_name):
    with open(file_name, 'rb') as f:
        d1 = np.loadtxt(f)
    print min(d1), max(d1)

    ax = plt.gca()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)
    ax.set_xticks([2500,5000,7500, 10000, 12500, 15000, 17500])
    ax.set_xticklabels([25,50,75,100,125,150,175])
    plt.xlim([0, 17500])
    plt.xlabel('Iterations, $10^3$', fontsize=20)
    plt.ylabel('Negative log-likelihood', fontsize=20)
    plt.plot(d1, 'r')
    plt.show()



s = '/mnt/sda4/CODING/python/kaggle_data/models/nfreq6featumeanlog_stdhighc180lowcu0.1win_l60strid60globa1recep[1, 2]use_t0activ[urelu, urelu, utanh]dropo[0.3, 0.6]overl9strid[1, 2]train10weigh0.01scale1nkern[32, 64, 512]pool_[1, 1]l2_re0.0001valid10max_i150000rando1'
s1 = s + '/output_1.0.txt'
s2 = s + '/output_0.01.txt'

plot_learning_curves(s1,s2)
plot_learning_curve(s1)
plot_learning_curve(s2)




# nrow = 10
# ncol = len(d1)/nrow
# d1 = d1[:ncol*nrow]
# d1 = np.reshape(d1, (nrow, ncol))
# df = DataFrame(data=d1)
# labels = xrange(0, ncol, 1000)
# plt.boxplot(df.values)
# plt.xticks(range(len(labels)), labels)
# plt.show()
#plt.plot(np.abs(d1[1:len(d1)]-d1[0:len(d1)-1]), 'r')
#plt.show()
