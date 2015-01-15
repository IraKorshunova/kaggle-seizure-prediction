import numpy as np
import json
import os
import cPickle
import copy

import matplotlib.pyplot as plt
from theano import config

from test_labels_loader.loader import load_train_data
from test_labels_loader.config_name_creator import *
from test_labels_loader.data_scaler import scale_across_features, scale_across_time


config.floatX = 'float32'


def plot_examples(subject, data_path, scale):
    d = load_train_data(data_path, subject)
    x, y, filename_to_idx = d['x'], d['y'], d['filename_to_idx']
    if scale:
        x, _ = scale_across_time(x=x)

    filename = 'Dog_1/Dog_1_interictal_segment_0001.mat'
    idx = filename_to_idx[filename]
    print filename_to_idx

    fig = plt.figure()
    fig.suptitle(filename)
    print x[idx].shape
    for i in range(x[idx].shape[0]):
        fig.add_subplot(4, 4, i)
        plt.imshow(x[idx, i, :, :], aspect='auto', origin='lower', interpolation='none')
        plt.colorbar()
    plt.show()

    # for filename, idx in filename_to_idx.items():
    #     fig = plt.figure()
    #     fig.suptitle(filename)
    #     for i in range(x[idx].shape[0]):
    #         fig.add_subplot(4, 4, i)
    #         plt.imshow(x[idx, i, :, :], aspect='auto', origin='lower', interpolation='none')
    #         plt.colorbar()
    #     plt.show()


if __name__ == '__main__':

    with open('SETTINGS.json') as f:
        settings_dict = json.load(f)

    # path
    data_path = settings_dict['path']['processed_data_path'] + '/' + create_fft_data_name(settings_dict)
    write_dir = data_path + '/img'
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)

    # params
    model_params = settings_dict['model']
    validation_params = settings_dict['validation']

    names = ['Dog_1', 'Dog_3', 'Dog_2', 'Dog_5', 'Dog_4', 'Patient_1', 'Patient_2']
    for subject in names:
        print '***********************', subject, '***************************'
        plot_examples(subject, data_path, scale=False)