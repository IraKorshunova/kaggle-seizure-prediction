import numpy as np

import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.metrics import confusion_matrix

from test_labels_loader import load_test_labels


def plot(clips, probs, labels):
    tx, sn, sp = [], [], []
    t_list = np.arange(0.0, 1.0, 0.02)
    for i, subject in enumerate(subjects):
        subject_idx = []
        for j, s in enumerate(clips):
            if subject in s:
                subject_idx.append(j)
        subject_idx = np.array(subject_idx)

        subj_prob = probs[subject_idx]
        y_true = labels[subject]['preictal']

        print '*******************', subject, '********************'
        y_pred = np.zeros_like(y_true)
        y_pred[np.where(subj_prob >= 0.5)] = 1
        cm = confusion_matrix(y_true, y_pred)
        sn_05 = 1.0 * cm[1, 1] / (cm[1, 1] + cm[1, 0])
        sp_05 = 1.0 * cm[0, 0] / (cm[0, 0] + cm[0, 1])
        print sn_05, sp_05

        subj_sn, subj_sp = [], []
        for t in t_list:
            y_pred = np.zeros_like(y_true)
            y_pred[np.where(subj_prob >= t)] = 1
            cm = confusion_matrix(y_true, y_pred)
            sn_t = 1.0 * cm[1, 1] / (cm[1, 1] + cm[1, 0])
            sp_t = 1.0 * cm[0, 0] / (cm[0, 0] + cm[0, 1])
            subj_sn.append(sn_t)
            subj_sp.append(sp_t)

        tx.append(t_list)
        sn.append(subj_sn)
        sp.append(subj_sp)

    for t_i, sn_i, sp_i, subject in zip(tx, sn, sp, subjects):
        plt.plot(t_i, sn_i, label=subject, linewidth=2.1)

    ax = plt.subplot(111)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
              ncol=4, fancybox=True, shadow=True, prop={'size': 20})
    plt.xlim([-0.01, 1.0])
    plt.ylim([- 0.01, 1.06])
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(25)
    # plt.ylabel('Specificity', fontsize=25)
    plt.ylabel('Sensitivity', fontsize=25)
    plt.xlabel('Threshold', fontsize=25)
    plt.show()


if __name__ == '__main__':
    test_labels_path = '/mnt/sda4/CODING/python/kaggle_data/test_labels.csv'
    submission_path = '/mnt/sda4/CODING/python/kaggle_data/submission_0.78612.csv'
    subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']

    submission_df = read_csv(submission_path)
    probs = submission_df['preictal']
    clips = submission_df['clip']
    labels = load_test_labels(test_labels_path)

    plot(clips, probs, labels)


