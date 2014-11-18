import numpy as np
import json
import os
import shutil
import cPickle
import preprocessors.fft as fft
from theano import config
from sklearn.cross_validation import StratifiedShuffleSplit
from cnn.conv_net import ConvNet
from utils.loader import load_grouped_train_data, load_train_data, load_test_data
from utils.config_name_creator import *
from utils.data_scaler import scale_across_time, scale_across_features
from utils.data_splitter import split_train_valid_filenames, generate_overlapped_data

config.floatX = 'float32'


def train(subject, data_path, model_path, model_params, validation_params):
    d = load_train_data(data_path, subject)
    x, y, filename_to_idx = d['x'], d['y'], d['filename_to_idx']
    x_test = load_test_data(data_path, subject)['x'] if model_params['use_test'] else None

    # --------- add params
    model_params['n_channels'] = x.shape[1]
    model_params['n_fbins'] = x.shape[2]
    model_params['n_timesteps'] = x.shape[3]

    print '============ parameters'
    for key, value in model_params.items():
        print key, ':', value
    print '========================'

    x_train, y_train = None, None
    x_valid, y_valid = None, None

    if model_params['overlap']:
        # no validation if overlap
        filenames_grouped_by_hour = cPickle.load(open('filenames.pickle'))
        data_grouped_by_hour = load_grouped_train_data(data_path, subject, filenames_grouped_by_hour)
        x, y = generate_overlapped_data(data_grouped_by_hour, overlap_size=model_params['overlap'],
                                        window_size=x.shape[-1],
                                        overlap_interictal=True,
                                        overlap_preictal=True)
        print x.shape

        x, scalers = scale_across_time(x, x_test=None) if model_params['scale_time'] \
            else scale_across_features(x, x_test=None)

        cnn = ConvNet(model_params)
        cnn.train(train_set=(x, y), max_iter=175000)
        state_dict = cnn.get_state()
        state_dict['scalers'] = scalers
        with open(model_path + '/' + subject + '.pickle', 'wb') as f:
            cPickle.dump(state_dict, f, protocol=cPickle.HIGHEST_PROTOCOL)
        return
    else:
        if validation_params['random_split']:
            skf = StratifiedShuffleSplit(y, n_iter=1, test_size=0.25, random_state=0)
            for train_idx, valid_idx in skf:
                x_train, y_train = x[train_idx], y[train_idx]
                x_valid, y_valid = x[valid_idx], y[valid_idx]
        else:
            filenames_grouped_by_hour = cPickle.load(open('filenames.pickle'))
            d = split_train_valid_filenames(subject, filenames_grouped_by_hour)
            train_filenames, valid_filenames = d['train_filenames'], d['valid_filnames']
            train_idx = [filename_to_idx[i] for i in train_filenames]
            valid_idx = [filename_to_idx[i] for i in valid_filenames]
            x_train, y_train = x[train_idx], y[train_idx]
            x_valid, y_valid = x[valid_idx], y[valid_idx]

    if model_params['scale_time']:
        x_train, scalers_train = scale_across_time(x=x_train, x_test=x_test)
        x_valid, _ = scale_across_time(x=x_valid, x_test=x_test, scalers=scalers_train)
    else:
        x_train, scalers_train = scale_across_features(x=x_train, x_test=x_test)
        x_valid, _ = scale_across_features(x=x_valid, x_test=x_test, scalers=scalers_train)

    del x, x_test

    print '============ dataset'
    print 'train:', x_train.shape
    print 'n_pos:', np.sum(y_train), 'n_neg:', len(y_train) - np.sum(y_train)
    print 'valid:', x_valid.shape
    print 'n_pos:', np.sum(y_valid), 'n_neg:', len(y_valid) - np.sum(y_valid)

    # -------------- validate
    cnn = ConvNet(model_params)
    best_iter = cnn.validate(train_set=(x_train, y_train),
                             valid_set=(x_valid, y_valid),
                             valid_freq=validation_params['valid_freq'],
                             max_iter=validation_params['max_iter'],
                             fname_out=model_path + '/' + subject + '.txt')

    # ---------------- scale
    d = load_train_data(data_path, subject)
    x, y, filename_to_idx = d['x'], d['y'], d['filename_to_idx']
    x_test = load_test_data(data_path, subject)['x'] if model_params['use_test'] else None

    x, scalers = scale_across_time(x=x, x_test=x_test) if model_params['scale_time'] \
        else scale_across_features(x=x, x_test=x_test)
    del x_test

    cnn = ConvNet(model_params)
    cnn.train(train_set=(x, y), max_iter=best_iter)
    state_dict = cnn.get_state()
    state_dict['scalers'] = scalers
    with open(model_path + '/' + subject + '.pickle', 'wb') as f:
        cPickle.dump(state_dict, f, protocol=cPickle.HIGHEST_PROTOCOL)


def run_trainer():
    with open('SETTINGS.json') as f:
        settings_dict = json.load(f)

    # path
    data_path = settings_dict['path']['processed_data_path'] + '/' + create_fft_data_name(settings_dict)
    model_path = settings_dict['path']['model_path'] + '/' + create_cnn_model_name(settings_dict)
    print data_path
    print model_path

    if not os.path.exists(data_path):
        fft.run_fft_preprocessor()

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copy2('SETTINGS.json', model_path + '/SETTINGS.json')

    # params
    model_params = settings_dict['model']
    validation_params = settings_dict['validation']

    subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    for subject in subjects:
        print '***********************', subject, '***************************'
        train(subject, data_path, model_path, model_params, validation_params)


if __name__ == '__main__':
    run_trainer()