import os, cPickle, copy, itertools
import re
import numpy as np
from scipy.io import loadmat
from collections import defaultdict


def load_grouped_train_data(data_path, subject, files_names_grouped_by_hour):
    def fill_data_grouped_by_hour(class_label, train_filenames):
        current_group_idx = 0
        current_ten_minutes_idx = 0
        for i, filename in enumerate(train_filenames):
            if subject + '/' + filename not in files_names_grouped_by_hour[subject][class_label][current_group_idx]:
                raise ValueError(
                    '{}/{} not in group for {}{}{}'.format(subject, filename, subject, class_label, current_group_idx))
            if current_ten_minutes_idx == 0:
                data_grouped_by_hour[class_label].append([])
            datum = loadmat(read_dir + '/' + filename, squeeze_me=True)
            data_grouped_by_hour[class_label][-1].append(datum['data'])
            current_ten_minutes_idx += 1
            if len(data_grouped_by_hour[class_label][-1]) == len(
                    files_names_grouped_by_hour[subject][class_label][current_group_idx]):
                current_group_idx += 1
                current_ten_minutes_idx = 0

    read_dir = data_path + '/' + subject
    filenames = sorted(os.listdir(read_dir))
    train_filenames = [filename for filename in filenames if 'test' not in filename]
    interictal_train_filenames = [filename for filename in train_filenames if 'interictal' in filename]
    preictal_train_filenames = [filename for filename in train_filenames if 'preictal' in filename]

    data_grouped_by_hour = defaultdict(lambda: [])
    fill_data_grouped_by_hour('interictal', interictal_train_filenames)
    fill_data_grouped_by_hour('preictal', preictal_train_filenames)

    return data_grouped_by_hour


def load_train_data(data_path, subject):
    read_dir = data_path + '/' + subject
    filenames = sorted(os.listdir(read_dir))
    train_filenames = []
    for filename in filenames:
        if 'test' not in filename:
            train_filenames.append(filename)

    n = len(train_filenames)
    datum = loadmat(read_dir + '/' + train_filenames[0], squeeze_me=True)
    x = np.zeros(((n,) + datum['data'].shape), dtype='float32')
    y = np.zeros(n, dtype='int8')

    filename_to_idx = {}
    for i, filename in enumerate(train_filenames):
        datum = loadmat(read_dir + '/' + filename, squeeze_me=True)
        x[i] = datum['data']
        y[i] = 1 if 'preictal' in filename else 0
        filename_to_idx[subject + '/' + filename] = i

    return {'x': x, 'y': y, 'filename_to_idx': filename_to_idx}


def load_test_data(data_path, subject):
    read_dir = data_path + '/' + subject
    data, id = [], []
    filenames = sorted(os.listdir(read_dir))
    for filename in sorted(filenames, key=lambda x: int(re.search(r'(\d+).mat', x).group(1))):
        if 'test' in filename or 'holdout' in filename:
            data.append(loadmat(read_dir + '/' + filename, squeeze_me=True))
            id.append(filename)

    n_test = len(data)
    x = np.zeros(((n_test,) + data[0]['data'].shape), dtype='float32')
    for i, datum in enumerate(data):
        x[i] = datum['data']

    return {'x': x, 'id': id}


def load_complete_channel_data(data_path, subject, channel_number):
    read_dir = data_path + '/' + subject
    filenames = sorted(os.listdir(read_dir))

    n = len(filenames)
    datum = loadmat(read_dir + '/' + filenames[0])
    x = np.zeros((n, datum['data'].shape[1], datum['data'].shape[2]), dtype='float32')
    y = np.zeros(n, dtype='int8')

    filename_to_idx = {}
    for i, filename in enumerate(filenames):
        datum = loadmat(read_dir + '/' + filename)
        x[i] = datum['data'][channel_number]
        if 'preictal' in filename:
            y[i] = 1
        elif 'interictal' in filename:
            y[i] = 0
        elif 'test' in filename:
            y[i] = -1
        filename_to_idx[subject + '/' + filename] = i

    return {'x': x, 'y':y, 'filename_to_idx': filename_to_idx}




