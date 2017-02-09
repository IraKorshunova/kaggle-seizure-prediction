import sys
import utils
import glob
import loader
import pathfinder
import numpy as np
from kaggle_auc import auc
from configuration import set_configuration


def evaluate_global_auc(predictions_exp_dir, holdout_subjects):
    clip2label, _ = loader.load_holdout_labels(pathfinder.LABELS_PATH)
    clip2prediction = {}
    prediction_files = [predictions_exp_dir + '/%s.pkl' % s for s in holdout_subjects]
    for p in prediction_files:
        id2prediction = utils.load_pkl(p)
        clip2prediction.update(id2prediction)

    targets, predictions = [], []
    for k, v in clip2prediction.iteritems():
        targets.append(clip2label[k])
        predictions.append(v)

    targets, predictions = np.array(targets), np.array(predictions)
    print
    print 'GLOBAL AUC:', auc(targets, predictions)
    print 'Sensitivity at 75% specificity:', utils.get_tpr(targets, predictions)


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

    half = last_test_idx if 'Dog_1' in holdout_subject else len(z) / 2
    print len(z[:half]), len(z[half:])
    return z[:half], z[half:]


def evaluate_dog1_dog4(predictions_exp_dir, holdout_subject):
    z1, _ = timesplit_test_holdout_predictions(predictions_exp_dir, holdout_subject)
    clips, times, predictions, labels, usage = zip(*z1)
    targets, predictions = np.array(labels), np.array(predictions)
    print len(predictions)
    print 'pp overlap Dog1 AUC:', auc(targets, predictions)


def evaluate_private_public(predictions_exp_dir, holdout_subject):
    clip2label_test, _, clip2usage = loader.load_test_labels(pathfinder.LABELS_PATH)
    subject_real = holdout_subject.replace('_holdout', '')
    prediction_file = predictions_exp_dir + '/%s-test.pkl' % subject_real
    clip2prediction = utils.load_pkl(prediction_file)

    targets, predictions = [], []
    for k, v in clip2prediction.iteritems():
        targets.append(clip2label_test[k])
        predictions.append(v)

    targets, predictions = np.array(targets), np.array(predictions)
    print
    print len(predictions)
    print 'pp AUC:', auc(targets, predictions)


def evaluate_public(predictions_exp_dir, holdout_subject):
    clip2label_test, _, clip2usage = loader.load_test_labels(pathfinder.LABELS_PATH)
    subject_real = holdout_subject.replace('_holdout', '')
    prediction_file = predictions_exp_dir + '/%s-test.pkl' % subject_real
    clip2prediction = utils.load_pkl(prediction_file)

    targets, predictions = [], []
    for k, v in clip2prediction.iteritems():
        if clip2usage[k] == 'Public':
            targets.append(clip2label_test[k])
            predictions.append(v)

    targets, predictions = np.array(targets), np.array(predictions)
    print
    print len(predictions)
    print 'public AUC:', auc(targets, predictions)


def evaluate_private(predictions_exp_dir, holdout_subject):
    clip2label_test, _, clip2usage = loader.load_test_labels(pathfinder.LABELS_PATH)
    subject_real = holdout_subject.replace('_holdout', '')
    prediction_file = predictions_exp_dir + '/%s-test.pkl' % subject_real
    clip2prediction = utils.load_pkl(prediction_file)

    targets, predictions = [], []
    for k, v in clip2prediction.iteritems():
        if clip2usage[k] == 'Private':
            targets.append(clip2label_test[k])
            predictions.append(v)

    targets, predictions = np.array(targets), np.array(predictions)
    print
    print len(predictions)
    print 'private AUC:', auc(targets, predictions)


def evaluate_private_public_holdout(predictions_exp_dir, holdout_subject):
    clip2label, _ = loader.load_holdout_labels(pathfinder.LABELS_PATH)
    clip2prediction = {}
    prediction_file = predictions_exp_dir + '/%s.pkl' % holdout_subject
    id2prediction = utils.load_pkl(prediction_file)
    clip2prediction.update(id2prediction)

    clip2label_test, _, clip2usage = loader.load_test_labels(pathfinder.LABELS_PATH)
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
    print len(predictions)
    print 'pph AUC:', auc(targets, predictions)


def d1(predictions_exp_dir, holdout_subject):
    z1, z2 = timesplit_test_holdout_predictions(predictions_exp_dir, holdout_subject)
    clips, times, predictions, labels, usage = zip(*z2)
    labels = list(labels)
    predictions = list(predictions)

    clip2label_test, _, clip2usage = loader.load_test_labels(pathfinder.LABELS_PATH)
    subject_real = holdout_subject.replace('_holdout', '')
    prediction_file = predictions_exp_dir + '/%s-test.pkl' % subject_real
    clip2prediction = utils.load_pkl(prediction_file)

    for k, v in clip2prediction.iteritems():
        labels.append(clip2label_test[k])
        predictions.append(v)

    targets, predictions = np.array(labels), np.array(predictions)
    print
    print len(predictions)
    print 'pp non overlap AUC:', auc(targets, predictions)



if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     sys.exit("Usage: evaluate.py <config_name>")

    # config_name = sys.argv[1]

    config_name = 'n8b1m_relu_sm'
    # config_name = 'lda_8b1m'
    # config_name = 'svm_rbf_c10'
    # config_name = 'svm_hrbf_c10'

    set_configuration(config_name)

    # predictions paths
    prediction_dir = utils.get_dir_path('predictions', pathfinder.METADATA_PATH)
    predictions_exp_dir = utils.find_model_metadata(prediction_dir, config_name)

    holdout_subjects = ['Dog_2_holdout', 'Dog_3_holdout']
    for s in holdout_subjects:
        print '==========', s, ' ==============='
        evaluate_private_public_holdout(predictions_exp_dir, s)
        evaluate_private_public(predictions_exp_dir, s)
        evaluate_public(predictions_exp_dir, s)
        evaluate_private(predictions_exp_dir, s)

    holdout_subjects = ['Dog_1_holdout', 'Dog_4_holdout']
    for s in holdout_subjects:
        print '==========', s, ' ==============='
        evaluate_dog1_dog4(predictions_exp_dir, s)
        evaluate_private_public_holdout(predictions_exp_dir, s)
        evaluate_private_public(predictions_exp_dir, s)
        d1(predictions_exp_dir,s)

    holdout_subjects = ['Dog_5', 'Patient_1', 'Patient_2', 'Dog_4']
    for s in holdout_subjects:
        print '==========', s, ' ==============='
        evaluate_private_public(predictions_exp_dir, s)
