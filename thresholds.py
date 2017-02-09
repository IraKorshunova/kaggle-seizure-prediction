import matplotlib
import sklearn

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np
import loader
import pathfinder
import configuration
import utils
import sklearn.metrics
import os


def thresholds(predictions_exp_dir, subjects):
    train_clip2label, _, _ = loader.load_test_labels(pathfinder.LABELS_PATH)
    for i, subject in enumerate(subjects):
        id2pred = utils.load_pkl(predictions_exp_dir + '/%s-test.pkl' % subject)
        subject_prob, subject_labels = [], []
        pp, ip = [], []
        for id, pred in id2pred.iteritems():
            subject_prob.append(pred)
            id = os.path.basename(id)
            subject_labels.append(train_clip2label[id])
            if train_clip2label[id] == 0:
                ip.append(pred)
            else:
                pp.append(pred)

        print subject
        fpr, tpr, threshold = sklearn.metrics.roc_curve(subject_labels, subject_prob)
        c = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
        opt_threshold = threshold[np.where(c == np.min(c))][0]
        print 'threshold', opt_threshold

        id2pred_test = utils.load_pkl(predictions_exp_dir + '/%s-test.pkl' % subject)
        test_clip2label, _, _ = loader.load_test_labels(pathfinder.LABELS_PATH)

        for k, v in id2pred_test.iteritems():
            id2pred_test[k] = 1 if v > opt_threshold else 0

        tp, fn, tn, fp = 0., 0., 0., 0.
        for k, v in id2pred_test.iteritems():
            if test_clip2label[k] == 1 and v == 1:
                tp += 1
            if test_clip2label[k] == 1 and v == 0:
                fn += 1
            if test_clip2label[k] == 0 and v == 0:
                tn += 1
            if test_clip2label[k] == 0 and v == 1:
                fp += 1

        sn = tp / (tp + fn)
        sp = tn / (tn + fp)
        print subject
        print 'Sn:', sn
        print 'SP:', sp


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit("Usage: evaluate.py <config_name>")

    config_name = sys.argv[1]
    configuration.set_configuration(config_name)
    prediction_dir = utils.get_dir_path('predictions', pathfinder.METADATA_PATH)
    predictions_exp_dir = utils.find_model_metadata(prediction_dir, config_name)
    subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    thresholds(predictions_exp_dir, subjects)
