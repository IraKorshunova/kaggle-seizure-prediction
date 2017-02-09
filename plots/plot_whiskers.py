import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from kaggle_auc import auc
from configuration import set_configuration
import sklearn.metrics
import scipy.stats.mstats
import utils
import sys
from configuration import set_configuration
import pathfinder
import loader


def evaluate_per_test_subject(submission_file, subject, threshold=0.5):
    clip2label, _, clip2usage = loader.load_test_labels(pathfinder.LABELS_PATH)
    prediction_file = predictions_exp_dir + '/%s-test.pkl' % subject
    clip2prediction = utils.load_pkl(prediction_file)

    tp, tn, fp, fn = 0, 0, 0, 0
    n_preictal, n_interictal = 0, 0
    targets, predictions = [], []
    for k, v in clip2prediction.iteritems():
        targets.append(clip2label[k])
        predictions.append(v)
        if clip2label[k] == 1:
            n_preictal += 1
        else:
            n_interictal += 1
        # print clip2label[k], v
        if clip2label[k] == 1 and v > threshold:
            tp += 1
        elif clip2label[k] == 1 and v < threshold:
            fn += 1
            # print 'fn', v
        elif clip2label[k] == 0 and v > threshold:
            fp += 1
            # print 'fp', v
        elif clip2label[k] == 0 and v < threshold:
            tn += 1
        else:
            raise ValueError()
    # print '# preictal', n_preictal
    # print '# interictal', n_interictal
    #
    # print 'TP', tp
    # print 'TN', tn
    # print 'FP', fp
    # print 'FN', fn
    #
    # targets, predictions = np.array(targets), np.array(predictions)
    # print subject, 'AUC:', auc(targets, predictions)
    # print subject, 'Sensitivity at 75% specificity:', utils.get_tpr(targets, predictions)
    # print subject, 'Sensitivity', 1.0 * tp / (tp + fn)
    # print subject, 'TIW', 1. * fp / n_interictal

    sn = 1.0 * tp / (tp + fn)
    fpr = 1. * fp / n_interictal
    return sn, fpr


def evaluate_per_holdout_subject(prediction_exp_dir, subject, threshold=0.5):
    clip2label, clip2time = loader.load_holdout_labels(pathfinder.LABELS_PATH)
    prediction_file = predictions_exp_dir + '/%s.pkl' % subject
    clip2prediction = utils.load_pkl(prediction_file)

    tp, tn, fp, fn = 0, 0, 0, 0
    n_preictal, n_interictal = 0, 0
    targets, predictions = [], []
    for k, v in clip2prediction.iteritems():
        targets.append(clip2label[k])
        predictions.append(v)
        if clip2label[k] == 1:
            n_preictal += 1
        else:
            n_interictal += 1
        # print clip2label[k], v
        if clip2label[k] == 1 and v > threshold:
            tp += 1
        elif clip2label[k] == 1 and v < threshold:
            fn += 1
            # print 'fn', v
        elif clip2label[k] == 0 and v > threshold:
            fp += 1
            # print 'fp', v
        elif clip2label[k] == 0 and v < threshold:
            tn += 1
        else:
            raise ValueError()
    # print '# preictal', n_preictal
    # print '# interictal', n_interictal
    #
    # print 'TP', tp
    # print 'TN', tn
    # print 'FP', fp
    # print 'FN', fn
    #
    # targets, predictions = np.array(targets), np.array(predictions)
    # try:
    #     print subject, 'AUC:', auc(targets, predictions)
    #     print subject, 'Sensitivity at 75% specificity:', utils.get_tpr(targets, predictions)
    #     print subject, 'Sensitivity', 1.0 * tp / (tp + fn)
    # except:
    #     pass
    #
    # print subject, 'FPR', 1. * fp / n_interictal
    sn = 1.0 * tp / (tp + fn)
    fpr = 1. * fp / n_interictal
    return sn, fpr


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: evaluate.py <config_name>")

    config_name = sys.argv[1]
    set_configuration(config_name)

    # predictions paths
    prediction_dir = utils.get_dir_path('predictions', pathfinder.METADATA_PATH)
    predictions_exp_dir = utils.find_model_metadata(prediction_dir, config_name)

    subjects_sn, subjects_sp = [], []
    subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4']
    for subject in subjects:
        print '================================', subject
        print evaluate_per_test_subject(predictions_exp_dir, subject)
        sn, sp = [], []
        for threshold in np.arange(0.2, 0.8, 0.01):
            sn1, sp1 = evaluate_per_test_subject(predictions_exp_dir, subject, threshold)
            sn.append(sn1)
            sp.append(sp1)
        subjects_sn.append(sn)
        subjects_sp.append(sp)


    plt.figure()
    plt.boxplot(subjects_sn)
    plt.savefig(pathfinder.IMG_PATH + '/%s-boxplot-sn.png' % config_name)

    plt.figure()
    plt.boxplot(subjects_sp)
    plt.savefig(pathfinder.IMG_PATH + '/%s-boxplot-sp.png' % config_name)

    # subjects = ['Dog_2_holdout', 'Dog_3_holdout', 'Dog_4_holdout']
    # for subject in subjects:
    #     evaluate_per_holdout_subject(predictions_exp_dir, subject)
    #     sn, sp = [], []
    #     for threshold in np.arange(0.2, 0.8, 0.01):
    #         sn1, sp1 = evaluate_per_holdout_subject(predictions_exp_dir, subject, threshold)
    #         sn.append(sn1)
    #         sp.append(sp1)
    #     print 'Sn', np.mean(sn), np.std(sn)
    #     print 'Sp', np.mean(sp), np.std(sp)
