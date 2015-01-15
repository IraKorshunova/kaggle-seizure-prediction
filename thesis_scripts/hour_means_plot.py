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


def plot_hour_means(x, filename_to_idx, hours, type, write_dir):
    for n_hour, hour in enumerate(hours):
        mean_fft = np.zeros((x.shape[1], x.shape[2], x.shape[3]))
        for clip in hour:
            idx = filename_to_idx[clip]
            mean_fft += x[idx]
        mean_fft /= len(hour)

        fig = plt.figure()
        fig.suptitle(type+str(n_hour))
        for i in range(mean_fft.shape[0]):
            fig.add_subplot(4, 4, i)
            plt.imshow(mean_fft[i, :, :], aspect='auto', origin='lower', interpolation='none')
            plt.colorbar()
        plt.show()
        plt.savefig(write_dir + '/' + subject + type + str(n_hour) + '.jpg')


def plot(subject, data_path, write_dir, scale):
    d = load_train_data(data_path, subject)
    x, y, filename_to_idx = d['x'], d['y'], d['filename_to_idx']
    if scale:
        x, _ = scale_across_time(x=x)

    filenames_grouped_by_hour = cPickle.load(open('filenames.pickle'))
    preictal_hours = copy.deepcopy(filenames_grouped_by_hour[subject]['preictal'])
    interictal_hours = copy.deepcopy(filenames_grouped_by_hour[subject]['interictal'])

    plot_hour_means(x, filename_to_idx, preictal_hours, type='preictal', write_dir=write_dir)
    plot_hour_means(x, filename_to_idx, interictal_hours[:10], type='interictal', write_dir=write_dir)


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
        plot(subject, data_path, write_dir, scale=False)