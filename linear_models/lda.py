import numpy as np
import json
import os

from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.lda import LDA

import preprocessors.fft as fft
from utils.loader import load_test_data, load_train_data
from utils.config_name_creator import *
from merger import merge_csv_files
from commons import reshape_data
from commons import load_test_labels


def train(subject, data_path, plot=False):
    d = load_train_data(data_path, subject)
    x, y = d['x'], d['y']
    print 'n_preictal', np.sum(y)
    print 'n_inetrictal', np.sum(y - 1)
    n_channels = x.shape[1]
    n_fbins = x.shape[2]

    x, y = reshape_data(x, y)
    data_scaler = StandardScaler()
    x = data_scaler.fit_transform(x)

    lda = LDA()
    lda.fit(x, y)
    coef = lda.scalings_ * lda.coef_[:1].T
    channels = []
    fbins = []
    for c in range(n_channels):
        fbins.extend(range(n_fbins))  # 0- delta, 1- theta ...
        channels.extend([c] * n_fbins)

    if plot:
        fig = plt.figure()
        for i in range(n_channels):
            if n_channels == 24:
                fig.add_subplot(4, 6, i)
            else:
                fig.add_subplot(4, 4, i)
            ax = plt.gca()
            ax.set_xlim([0, n_fbins])
            ax.set_xticks(np.arange(0.5, n_fbins + 0.5, 1))
            ax.set_xticklabels(np.arange(0, n_fbins))
            max_y = max(abs(coef)) + 0.01
            ax.set_ylim([0, max_y])
            ax.set_yticks(np.around(np.arange(0, max_y, max_y / 4.0), decimals=1))
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(15)
            plt.bar(range(0, n_fbins), abs(coef[i * n_fbins:i * n_fbins + n_fbins]))
        fig.suptitle(subject, fontsize=20)
        plt.show()

    coefs = np.reshape(coef, (n_channels, n_fbins))
    return lda, data_scaler, coefs


def predict(subject, model, data_scaler, data_path, submission_path, test_labels, opt_threshold_train):
    d = load_test_data(data_path, subject)
    x_test, id = d['x'], d['id']
    n_test_examples = x_test.shape[0]
    n_timesteps = x_test.shape[3]

    x_test = reshape_data(x_test)
    x_test = data_scaler.transform(x_test)

    pred_1m = model.predict_proba(x_test)[:, 1]

    pred_10m = np.reshape(pred_1m, (n_test_examples, n_timesteps))
    pred_10m = np.mean(pred_10m, axis=1)
    ans = zip(id, pred_10m)
    df = DataFrame(data=ans, columns=['clip', 'preictal'])
    df.to_csv(submission_path + '/' + subject + '.csv', index=False, header=True)


def run_trainer():
    with open('SETTINGS.json') as f:
        settings_dict = json.load(f)

    data_path = settings_dict['path']['processed_data_path'] + '/' + create_fft_data_name(settings_dict)
    submission_path = settings_dict['path']['submission_path'] + '/LDA_' + create_fft_data_name(settings_dict)
    print data_path

    if not os.path.exists(data_path):
        fft.run_fft_preprocessor()

    if not os.path.exists(submission_path):
        os.makedirs(submission_path)

    test_labels_path = '/mnt/sda4/CODING/python/kaggle_data/test_labels.csv'
    test_labels = load_test_labels(test_labels_path)

    subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    coef_list = []
    for subject in subjects:
        print '***********************', subject, '***************************'
        model, data_scaler, coefs  = train(subject, data_path)
        predict(subject, model, data_scaler, data_path, submission_path, test_labels[subject]['preictal'])
        coef_list.append(coefs)

    merge_csv_files(submission_path, subjects, 'submission')
    merge_csv_files(submission_path, subjects, 'submission_softmax')
    merge_csv_files(submission_path, subjects, 'submission_minmax')
    merge_csv_files(submission_path, subjects, 'submission_median')


if __name__ == '__main__':
    run_trainer()