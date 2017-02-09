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


def find_threshold(sp, thresholds, sp_level=75):
    diffs = np.abs(np.array(sp) - sp_level)
    w = np.argwhere(diffs == np.min(diffs))
    w_idxs = w.flatten().tolist()
    mean_t = []
    for i in w_idxs:
        mean_t.append(thresholds[i])
    return np.mean(mean_t)


def evaluate_cv(predictions_exp_dir, subject, threshold):
    predictions_path = predictions_exp_dir + '/' + subject + '-cv.pkl'
    d = utils.load_pkl(predictions_path)
    predictions, targets = d['predictions'], d['targets']

    tp, tn, fp, fn = 0, 0, 0, 0
    n_preictal, n_interictal = 0, 0
    for p, t in zip(predictions, targets):
        if t == 1:
            n_preictal += 1
        else:
            n_interictal += 1
        if t == 1 and p >= threshold:
            tp += 1
        elif t == 1 and p < threshold:
            fn += 1
            # print 'fn', v
        elif t == 0 and p >= threshold:
            fp += 1
            # print 'fp', v
        elif t == 0 and p < threshold:
            tn += 1
        else:
            print p, t
            raise ValueError()
    # print '# preictal', n_preictal
    # print '# interictal', n_interictal
    # print 'TP', tp
    # print 'TN', tn
    # print 'FP', fp
    # print 'FN', fn
    # targets, predictions = np.array(targets), np.array(predictions)

    print subject, 'AUC:', auc(targets, predictions)
    sn = 1.0 * tp / n_preictal * 100 if n_preictal > 0 else None
    sp = 1. * tn / n_interictal * 100
    ppv = 1. * tp / (tp + fp) * 100 if tp + fp > 0 else None
    npv = 1. * tn / (tn + fn) * 100
    return sn, sp, ppv, npv


def evaluate_per_test_holdout_subject(predictions_exp_dir, holdout_subject):
    clip2label, _ = loader.load_holdout_labels(pathfinder.LABELS_PATH)
    clip2prediction = {}
    prediction_file = predictions_exp_dir + '/%s.pkl' % holdout_subject
    id2prediction = utils.load_pkl(prediction_file)
    clip2prediction.update(id2prediction)

    clip2label_test, _, _ = loader.load_test_labels(pathfinder.LABELS_PATH)
    subject_real = holdout_subject.replace('_holdout', '')
    prediction_file = predictions_exp_dir + '/%s-test.pkl' % subject_real
    id2prediction = utils.load_pkl(prediction_file)
    clip2prediction.update(id2prediction)

    clip2label.update(clip2label_test)

    targets, predictions = [], []
    for k, v in clip2prediction.iteritems():
        targets.append(clip2label[k])
        predictions.append(v)

    targets, predictions = np.array(targets), np.array(predictions)
    print
    print 'AUC:', auc(targets, predictions)
    print 'Sensitivity at 75% specificity:', utils.get_tpr(targets, predictions)


def evaluate_per_train_subject(submission_file, subject, threshold=0.5):
    clip2label = loader.load_train_labels(pathfinder.LABELS_PATH)
    prediction_file = predictions_exp_dir + '/%s-train.pkl' % subject
    clip2prediction = utils.load_pkl(prediction_file)

    tp, tn, fp, fn = 0, 0, 0, 0
    n_preictal, n_interictal = 0, 0
    targets, predictions = [], []
    for k, v in clip2prediction.iteritems():
        k = k.split('/')[-1]
        targets.append(clip2label[k])
        predictions.append(v)
        if clip2label[k] == 1:
            n_preictal += 1
        else:
            n_interictal += 1
        if clip2label[k] == 1 and v >= threshold:
            tp += 1
        elif clip2label[k] == 1 and v < threshold:
            fn += 1
        elif clip2label[k] == 0 and v >= threshold:
            fp += 1
        elif clip2label[k] == 0 and v < threshold:
            tn += 1
        else:
            print clip2label[k], v
            raise ValueError()
    # print '# preictal', n_preictal
    # print '# interictal', n_interictal
    # print 'TP', tp
    # print 'TN', tn
    # print 'FP', fp
    # print 'FN', fn
    # targets, predictions = np.array(targets), np.array(predictions)
    # print subject, 'AUC:', auc(targets, predictions)
    sn = 1.0 * tp / n_preictal * 100 if n_preictal > 0 else None
    sp = 1. * tn / n_interictal * 100
    ppv = 1. * tp / (tp + fp) * 100 if tp + fp > 0 else None
    npv = 1. * tn / (tn + fn) * 100 if tn + fn > 0 else None
    return sn, sp, ppv, npv


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
        if clip2label[k] == 1 and v >= threshold:
            tp += 1
        elif clip2label[k] == 1 and v < threshold:
            fn += 1
            # print 'fn', v
        elif clip2label[k] == 0 and v >= threshold:
            fp += 1
            # print 'fp', v
        elif clip2label[k] == 0 and v < threshold:
            tn += 1
        else:
            print clip2label[k], v
            raise ValueError()
    # print '# preictal', n_preictal
    # print '# interictal', n_interictal
    # print 'TP', tp
    # print 'TN', tn
    # print 'FP', fp
    # print 'FN', fn
    # targets, predictions = np.array(targets), np.array(predictions)
    # print subject, 'AUC:', auc(targets, predictions)
    sn = 1.0 * tp / n_preictal * 100 if n_preictal > 0 else None
    sp = 1. * tn / n_interictal * 100
    ppv = 1. * tp / (tp + fp) * 100 if tp + fp > 0 else None
    npv = 1. * tn / (tn + fn) * 100 if tn + fn > 0 else None
    return sn, sp, ppv, npv


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
        if clip2label[k] == 1 and v >= threshold:
            tp += 1
        elif clip2label[k] == 1 and v < threshold:
            fn += 1
            # print 'fn', v
        elif clip2label[k] == 0 and v >= threshold:
            fp += 1
            # print 'fp', v
        elif clip2label[k] == 0 and v < threshold:
            tn += 1
        else:
            raise ValueError()
    # print '# preictal', n_preictal
    # print '# interictal', n_interictal
    # print 'TP', tp
    # print 'TN', tn
    # print 'FP', fp
    # print 'FN', fn
    # targets, predictions = np.array(targets), np.array(predictions)
    # print subject, 'AUC:', auc(targets, predictions)

    sn = 1.0 * tp / n_preictal * 100 if n_preictal > 0 else None
    sp = 1. * tn / n_interictal * 100
    ppv = 1. * tp / (tp + fp) * 100 if tp + fp > 0 else None
    npv = 1. * tn / (tn + fn) * 100 if tn + fn > 0 else None
    return sn, sp, ppv, npv


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: evaluate.py <config_name>")

    config_name = sys.argv[1]
    set_configuration(config_name)

    # predictions paths
    prediction_dir = utils.get_dir_path('predictions', pathfinder.METADATA_PATH)
    predictions_exp_dir = utils.find_model_metadata(prediction_dir, config_name)

    thresholds = np.arange(0.0, 1.0, 0.01)
    d = {'thresholds': thresholds, 'sn_test': {}, 'sp_test': {}, 'sn_holdout': {}, 'sp_holdout': {},
         't_train': {}, 't_test': {}, 't_holdout': {}, 't_test_test': {},
         't_test_holdout': {}, 't_holdout_test': {}, 't_holdout_holdout': {}}

    subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4']

    for subject in subjects:
        print '================================ TR', subject
        sn, sp = [], []
        tsp75 = []
        for t in thresholds:
            sn_t, sp_t, _, _ = evaluate_per_train_subject(predictions_exp_dir, subject, t)
            sn.append(sn_t)
            sp.append(sp_t)
            tsp75.append(t)
        tsp75 = find_threshold(sp, tsp75)
        d['t_train'][subject] = tsp75

        print 'AT 75 % SP train, threshold=', tsp75
        print evaluate_per_train_subject(predictions_exp_dir, subject, tsp75)
        print 'AT 75 % SP test, threshold=', tsp75
        print evaluate_per_test_subject(predictions_exp_dir, subject, tsp75)
        print 'AT 75 % SP holdout, threshold=', tsp75
        print evaluate_per_holdout_subject(predictions_exp_dir, subject + '_holdout', tsp75)

    print '#######################################################################################'
    for subject in subjects:
        print '================================ TR', subject
        sn, sp, thresholds = [], [], np.arange(0.0, 1.0, 0.01)
        tsp75 = []
        for t in thresholds:
            sn_t, sp_t, _, _ = evaluate_per_test_subject(predictions_exp_dir, subject, t)
            sn.append(sn_t)
            sp.append(sp_t)
            tsp75.append(t)
        tsp75 = find_threshold(sp, tsp75)

        d['sn_test'][subject] = sn
        d['sp_test'][subject] = sp
        d['t_test'][subject] = tsp75
        d['t_test_test'][subject] = evaluate_per_test_subject(predictions_exp_dir, subject, tsp75)
        d['t_test_holdout'][subject] = evaluate_per_holdout_subject(predictions_exp_dir, subject + '_holdout', tsp75)

        print 'AT 75 % SP test, threshold=', tsp75
        print evaluate_per_test_subject(predictions_exp_dir, subject, tsp75)
        print 'AT 75 % SP holdout, threshold=', tsp75
        print evaluate_per_holdout_subject(predictions_exp_dir, subject + '_holdout', tsp75)

    subjects = ['Dog_1_holdout', 'Dog_2_holdout', 'Dog_3_holdout', 'Dog_4_holdout']
    for subject in subjects:
        print '================================ TR', subject
        sn, sp, thresholds = [], [], np.arange(0.0, 1.0, 0.01)
        tsp75 = []
        for t in thresholds:
            sn_t, sp_t, _, _ = evaluate_per_holdout_subject(predictions_exp_dir, subject, t)
            sn.append(sn_t)
            sp.append(sp_t)
            tsp75.append(t)

        tsp75 = find_threshold(sp, tsp75)

        d['sn_holdout'][subject] = sn
        d['sp_holdout'][subject] = sp
        d['t_holdout'][subject] = tsp75
        d['t_holdout_test'][subject] = evaluate_per_test_subject(predictions_exp_dir, subject.replace('_holdout',''), tsp75)
        d['t_holdout_holdout'][subject] = evaluate_per_holdout_subject(predictions_exp_dir, subject, tsp75)

        print 'AT 75% SP holdout, threshold=', tsp75
        print evaluate_per_holdout_subject(predictions_exp_dir, subject, tsp75)

    utils.save_pkl(d, pathfinder.IMG_PATH + '/%s-sp_sn.pkl' % config_name)
