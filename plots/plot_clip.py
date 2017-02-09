import sys
import numpy as np
import theano
import lasagne as nn
import utils
from configuration import config, set_configuration
import pathfinder
import data_iterators
import string
import glob
import csv
import kaggle_auc
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def predict_train_test(subject):
    print 'Load data'
    train_data_iterator = data_iterators.DataGenerator(subject=subject, batch_size=config().batch_size,
                                                       dataset='train',
                                                       transform_params=config().transformation_params,
                                                       full_batch=False, random=False, infinite=False)

    for xs_batch, _, _ in train_data_iterator.generate():
        x = xs_batch[0]
        print x.shape
        ax = plt.gca()
        # bands = np.arange(0, 181, 30) * x.shape[-1] / (sampling_frequency) / 10
        # ax.set_yticks(bands)
        # ax.set_yticklabels(np.arange(0, 181, 30))
        ax.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelbottom='off')  # labels along the bottom edge are off
        plt.setp(ax.get_yticklabels(), visible=False)
        # plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xticks(range(0, 10))
        ax.set_xticklabels(range(1, 11))
        for label in (ax.get_xticklabels()):
            label.set_fontsize(35)
        plt.imshow(x[0, :, :], aspect='auto', origin='lower', interpolation='none')
        # cbar = plt.colorbar()
        # cbar.ax.tick_params(labelsize=20)
        plt.xlabel('Time, min', fontsize=55)
        plt.ylabel('Channel 1', fontsize=55)
        plt.tight_layout()
        plt.savefig(pathfinder.IMG_PATH + '/%s-clip0.png' % config_name)
        return


if __name__ == '__main__':
    config_name = 'n8b1m_relu_sm'
    set_configuration(config_name)

    predict_train_test('Dog_1')
