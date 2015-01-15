import numpy as np
import json
import os

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.lda import LDA
from sklearn.metrics import confusion_matrix

import preprocessors.fft as fft
from utils.loader import load_test_data, load_train_data
from utils.config_name_creator import *
from commons import reshape_data
from commons import load_test_labels
from commons import print_cm
from sklearn.metrics import roc_curve


def curve_per_subject(subject, data_path, test_labels):
    d = load_train_data(data_path, subject)
    x, y_10m = d['x'], d['y']
    n_train_examples = x.shape[0]
    n_timesteps = x.shape[-1]
    print 'n_preictal', np.sum(y_10m)
    print 'n_inetrictal', np.sum(y_10m - 1)

    x, y = reshape_data(x, y_10m)
    data_scaler = StandardScaler()
    x = data_scaler.fit_transform(x)

    lda = LDA()
    lda.fit(x, y)

    pred_1m = lda.predict_proba(x)[:, 1]
    pred_10m = np.reshape(pred_1m, (n_train_examples, n_timesteps))
    pred_10m = np.mean(pred_10m, axis=1)
    fpr, tpr, threshold = roc_curve(y_10m, pred_10m)
    c = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
    opt_threshold = threshold[np.where(c == np.min(c))[0]][-1]
    print opt_threshold

    # ------- TEST ---------------

    d = load_test_data(data_path, subject)
    x_test, id = d['x'], d['id']
    n_test_examples = x_test.shape[0]
    n_timesteps = x_test.shape[3]
    x_test = reshape_data(x_test)
    x_test = data_scaler.transform(x_test)

    pred_1m = lda.predict_proba(x_test)[:, 1]
    pred_10m = np.reshape(pred_1m, (n_test_examples, n_timesteps))
    pred_10m = np.mean(pred_10m, axis=1)

    y_pred = np.zeros_like(test_labels)
    y_pred[np.where(pred_10m >= opt_threshold)] = 1
    cm = confusion_matrix(test_labels, y_pred)
    print print_cm(cm, labels=['interictal', 'preictal'])
    sn = 1.0 * cm[1, 1] / (cm[1, 1] + cm[1, 0])
    sp = 1.0 * cm[0, 0] / (cm[0, 0] + cm[0, 1])
    print sn, sp

    sn, sp = [], []
    t_list = np.arange(0.0, 1.0, 0.01)
    for t in t_list:
        y_pred = np.zeros_like(test_labels)
        y_pred[np.where(pred_10m >= t)] = 1
        cm = confusion_matrix(test_labels, y_pred)
        sn_t = 1.0 * cm[1, 1] / (cm[1, 1] + cm[1, 0])
        sp_t = 1.0 * cm[0, 0] / (cm[0, 0] + cm[0, 1])
        sn.append(sn_t)
        sp.append(sp_t)

    return t_list, sn, sp


if __name__ == '__main__':
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
    t, sn, sp = [], [], []
    for subject in subjects:
        print '***********************', subject, '***************************'
        t_i, sn_i, sp_i = curve_per_subject(subject, data_path, test_labels[subject]['preictal'])
        t.append(t_i)
        sn.append(sn_i)
        sp.append(sp_i)

    ax = plt.subplot(111)
    plt.xlim([-0.01, 1.0])
    plt.ylim([- 0.01, 1.06])
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(25)
    plt.xlabel('Threshold', fontsize=25)

    # plt.ylabel('Sensitivity', fontsize=25)
    # for t_i, sn_i, subject in zip(t, sn, subjects):
    #     plt.plot(t_i, sn_i, label=subject, linewidth=2.0)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
    #           ncol=4, fancybox=True, shadow=True, prop={'size': 20})
    # plt.show()

    plt.ylabel('Specificity', fontsize=25)
    for t_i, sp_i, subject in zip(t, sp, subjects):
        plt.plot(t_i, sp_i, label=subject, linewidth=2.0)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
              ncol=4, fancybox=True, shadow=True, prop={'size': 20})
    plt.show()