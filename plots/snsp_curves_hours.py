import matplotlib

matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import utils
import sys
from configuration import set_configuration
import pathfinder
import loader


def plot(subjects, sp_curve=True):
    tx, sn, sp = [], [], []
    t_list = np.arange(0.0, 1.0, 0.02)
    for i, subject in enumerate(subjects):
        clip2label, _, _ = loader.load_test_labels(pathfinder.LABELS_PATH)
        prediction_file = predictions_exp_dir + '/%s-test.pkl' % subject
        clip2prediction = utils.load_pkl(prediction_file)

        subj_sn, subj_sp = [], []
        y_true = []
        y_pred = []
        for k, v in clip2prediction.iteritems():
            y_true.append(clip2label[k])
            y_pred.append(v)
        for t in t_list:
            y_pred_bin = [1 if p >= t else 0 for p in y_pred]
            cm = confusion_matrix(y_true, y_pred_bin)
            sn_t = 1.0 * cm[1, 1] / (cm[1, 1] + cm[1, 0])
            sp_t = 1.0 * cm[0, 0] / (cm[0, 0] + cm[0, 1])
            subj_sn.append(sn_t)
            subj_sp.append(sp_t)

        tx.append(t_list)
        sn.append(subj_sn)
        sp.append(subj_sp)

    plt.figure()
    c = ['Purple', 'Red', 'DarkOliveGreen', 'MediumBlue']
    for t_i, sn_i, sp_i, subject, col in zip(tx, sn, sp, subjects, c):
        if sp_curve:
            plt.plot(t_i, sp_i, label=subject, linewidth=3.0, color=col)
        else:
            plt.plot(t_i, sn_i, label=subject, linewidth=3.0, color=col)

    ax = plt.subplot(111)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
    #           ncol=5, fancybox=True, shadow=True, prop={'size': 25})
    plt.xlim([-0.0, 1.0])
    plt.ylim([- 0.0, 1.1])
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(25)
    if sp_curve:
        plt.ylabel('Specificity', fontsize=25)
    else:
        plt.ylabel('Sensitivity', fontsize=25)
    plt.xlabel('Threshold', fontsize=25)
    plt.tight_layout()
    if sp_curve:
        plt.savefig(pathfinder.IMG_PATH + '/%s-sp.png' % config_name)
    else:
        plt.savefig(pathfinder.IMG_PATH + '/%s-sn.png' % config_name)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit("Usage: evaluate.py <config_name>")

    config_name = sys.argv[1]
    set_configuration(config_name)

    # predictions paths
    prediction_dir = utils.get_dir_path('predictions', pathfinder.METADATA_PATH)
    predictions_exp_dir = utils.find_model_metadata(prediction_dir, config_name)

    subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4']
    plot(subjects, True)
    plot(subjects, False)
