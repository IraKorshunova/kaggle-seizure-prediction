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


def find_opt_cv_threshold(prediction_exp_dir, subject):
    predictions_path = predictions_exp_dir + '/' + subject + '-cv.pkl'
    d = utils.load_pkl(predictions_path)
    predictions, targets = d['predictions'], d['targets']
    fpr, tpr, threshold = sklearn.metrics.roc_curve(targets, predictions)
    c = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
    opt_threshold = threshold[np.where(c == np.min(c))][0]
    return opt_threshold


def find_opt_test_threshold(test_set):
    clips, times, predictions, labels, usage = zip(*test_set)
    try:
        fpr, tpr, threshold = sklearn.metrics.roc_curve(labels, predictions)
        c = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
        opt_threshold = threshold[np.where(c == np.min(c))][0]
    except:
        opt_threshold = 0.0
    return opt_threshold


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
        elif t == 0 and p >= threshold:
            fp += 1
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

    # try:
    #     print subject, 'AUC:', auc(targets, predictions)
    # except:
    #     print 'no auc'
    sn = 1.0 * tp / n_preictal * 100 if n_preictal > 0 else None
    sp = 1. * tn / n_interictal * 100
    ppv = 1. * tp / (tp + fp) * 100 if tp + fp > 0 else None
    npv = 1. * tn / (tn + fn) * 100 if tn + fn > 0 else None

    return sn, sp, ppv, npv


def timesplit_test_holdout_predictions(predictions_exp_dir, holdout_subject):
    clip2label, clip2time = loader.load_holdout_labels(pathfinder.LABELS_PATH)
    prediction_file = predictions_exp_dir + '/%s.pkl' % holdout_subject
    clip2prediction = utils.load_pkl(prediction_file)
    hclips, htimes, hpredictions, hlabels = [], [], [], []
    print '==============================='

    for k, v in clip2prediction.iteritems():
        hpredictions.append(v)
        hclips.append(k)
        hlabels.append(clip2label[k])
        htimes.append(clip2time[k])
    usage = ['holdout'] * len(hclips)

    clip2label, clip2time, _ = loader.load_test_labels(pathfinder.LABELS_PATH)
    subject_real = holdout_subject.replace('_holdout', '')
    prediction_file = predictions_exp_dir + '/%s-test.pkl' % subject_real
    clip2prediction = utils.load_pkl(prediction_file)

    tclips, ttimes, tpredictions, tlabels = [], [], [], []

    for k, v in clip2prediction.iteritems():
        tpredictions.append(v)
        tclips.append(k)
        tlabels.append(clip2label[k])
        ttimes.append(clip2time[k])
        usage.append('test')

    clips = hclips + tclips
    times = htimes + ttimes
    predictions = hpredictions + tpredictions
    labels = hlabels + tlabels

    z = zip(clips, times, predictions, labels, usage)
    z.sort(key=lambda x: x[1])
    last_test_idx = 0
    for i, (c, t, p, l, u) in enumerate(z):
        # print c, t, p, l, u
        if u == 'test':
            last_test_idx = i

    half = last_test_idx if 'Dog_1' in holdout_subject or 'Dog_4' in holdout_subject else len(z)
    print len(z[:half]), len(z[half:])
    return z[:half], z[half:]


def evaluate_auc_cv_subject(predictions_exp_dir, subject):
    predictions_path = predictions_exp_dir + '/' + subject + '-cv.pkl'
    d = utils.load_pkl(predictions_path)
    predictions, targets = d['predictions'], d['targets']
    targets, predictions = np.array(targets), np.array(predictions)
    try:
        print subject, 'AUC:', auc(targets, predictions)
    except:
        print 'no auc'


def evaluate_auc_test_subject(test_set):
    clips, times, predictions, labels, usage = zip(*test_set)
    targets, predictions = np.array(labels), np.array(predictions)
    try:
        print subject, 'AUC:', auc(targets, predictions)
    except:
        print 'no auc'


def evaluate_per_test_subject(test_set, threshold=0.5):
    if len(test_set) == 0:
        return 0,0,0,0
    clips, times, predictions, labels, usage = zip(*test_set)
    tp, tn, fp, fn = 0, 0, 0, 0
    n_preictal, n_interictal = 0, 0
    clip2prediction, clip2label = {}, {}

    for c, p, l in zip(clips, predictions, labels):
        clip2prediction[c] = p
        clip2label[c] = l

    for k, v in clip2prediction.iteritems():
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
    # try:
    #     print subject, 'AUC:', auc(labels, predictions)
    # except:
    #     'no AUC'
    sn = 1.0 * tp / n_preictal * 100 if n_preictal > 0 else None
    sp = 1. * tn / n_interictal * 100
    ppv = 1. * tp / (tp + fp) * 100 if tp + fp > 0 else None
    npv = 1. * tn / (tn + fn) * 100 if tn + fn > 0 else None
    return sn, sp, ppv, npv


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     sys.exit("Usage: evaluate.py <config_name>")

    # config_name = sys.argv[1]

    # config_name = 'n8b1m_relu_sm'
    # config_name = 'lda_8b1m'
    # config_name = 'svm_rbf_c10'
    # config_name = 'svm_hrbf_c10'
    config_name = 'h8b1m'


    # predictions paths
    prediction_dir = utils.get_dir_path('predictions', pathfinder.METADATA_PATH)
    predictions_exp_dir = utils.find_model_metadata(prediction_dir, config_name)

    thresholds = np.arange(0.0, 1.0, 0.01)
    d = {'thresholds': thresholds, 'sn_test': {}, 'sp_test': {}, 'sn_holdout': {}, 'sp_holdout': {},
         'ppv_test': {}, 'ppv_holdout': {},
         't_test': {}, 't_holdout': {}, 't_cv75': {}, 't_cv_opt': {}}

    subjects = ['Dog_1_holdout', 'Dog_2_holdout', 'Dog_3_holdout', 'Dog_4_holdout']
    # subjects = ['Dog_3_holdout']
    for subject in subjects:
        print subject
        testset1, testset2 = timesplit_test_holdout_predictions(predictions_exp_dir, subject)
        # print 'CV threshold 0.5', evaluate_cv(predictions_exp_dir, subject.replace('_holdout', ''), 0.5)
        print 'Set 1 threshold 0.5', evaluate_per_test_subject(testset1, 0.5)
        print 'Set 2 threshold 0.5', evaluate_per_test_subject(testset2, 0.5)

        # if 'lda' in config_name or 'svm' in config_name:
        #     print '================================ TR optimized on CV', subject
        #     sn, sp = [], []
        #     tsp75 = []
        #     for t in thresholds:
        #         sn_t, sp_t, _, _, = evaluate_cv(predictions_exp_dir, subject.replace('_holdout', ''), t)
        #         sn.append(sn_t)
        #         sp.append(sp_t)
        #         tsp75.append(t)
        #     tsp75 = find_threshold(sp, tsp75)
        #     d['t_cv75'][subject] = tsp75
        #     print '75 t=', tsp75
        #     print 'cv-cv', evaluate_cv(predictions_exp_dir, subject.replace('_holdout', ''), tsp75)
        #     print 'cv-test1', evaluate_per_test_subject(testset1, tsp75)
        #     print 'cv-test2', evaluate_per_test_subject(testset2, tsp75)
        #
        #     topt = find_opt_cv_threshold(predictions_exp_dir, subject.replace('_holdout', ''))
        #     d['t_cv_opt'][subject] = topt
        #     print 'opt t=', topt
        #     print 'opt cv-cv', evaluate_cv(predictions_exp_dir, subject.replace('_holdout', ''), topt)
        #     print 'opt cv-test1', evaluate_per_test_subject(testset1, topt)
        #     print 'opt cv-test2', evaluate_per_test_subject(testset2, topt)

        print '================================ TR optimized on set 1', subject
        sn, sp, ppv, thresholds = [], [], [], np.arange(0.0, 1.0, 0.01)
        tsp75 = []
        for t in thresholds:
            sn_t, sp_t, ppv_t, _ = evaluate_per_test_subject(testset1, t)
            sn.append(sn_t)
            sp.append(sp_t)
            ppv.append(ppv_t)
            tsp75.append(t)

        tsp75 = find_threshold(sp, tsp75)

        d['sn_test'][subject] = sn
        d['sp_test'][subject] = sp
        d['ppv_test'][subject] = ppv
        # d['t_test'][subject] = tsp75
        # d['t_test_test'][subject] = evaluate_per_test_subject(testset1, tsp75)
        # d['t_test_holdout'][subject] = evaluate_per_test_subject(testset2, tsp75)
        # print tsp75
        # print 'test1-test1', d['t_test_test'][subject]
        # print 'test1-test2', d['t_test_holdout'][subject]
        #
        # topt = find_opt_test_threshold(testset1)
        # print 'opt test1-test1', evaluate_per_test_subject(testset1, topt)
        # print 'opt test1-test2', evaluate_per_test_subject(testset2, topt)

        print '================================ TR optimized on set 2', subject
        sn, sp, ppv, thresholds = [], [], [], np.arange(0.0, 1.0, 0.01)
        tsp75 = []
        for t in thresholds:
            sn_t, sp_t, ppv_t, _ = evaluate_per_test_subject(testset2, t)
            sn.append(sn_t)
            sp.append(sp_t)
            ppv.append(ppv_t)
            tsp75.append(t)

        tsp75 = find_threshold(sp, tsp75)

        d['sn_holdout'][subject] = sn
        d['sp_holdout'][subject] = sp
        d['ppv_holdout'][subject] = ppv
        # d['t_holdout'][subject] = tsp75
        # d['t_holdout_test'][subject] = evaluate_per_test_subject(testset1, tsp75)
        # d['t_holdout_holdout'][subject] = evaluate_per_test_subject(testset2, tsp75)
        # print tsp75
        # print 'test2-test2', d['t_holdout_holdout'][subject]
        # print 'test2-test1', d['t_holdout_test'][subject]
        #
        # topt = find_opt_test_threshold(testset2)
        # print 'opt test2-test1', evaluate_per_test_subject(testset1, topt)
        # print 'opt test2-test2', evaluate_per_test_subject(testset2, topt)

    utils.save_pkl(d, pathfinder.IMG_PATH + '/%s-sp_sn.pkl' % config_name)
