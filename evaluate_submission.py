import sys
import utils
import glob
import csv
import loader
import pathfinder
import numpy as np
from kaggle_auc import auc
from configuration import set_configuration
import sklearn.metrics


def tp_per_seizure(submission_file, subject, threshold):
    clip2prediction = {}
    with open(submission_file, 'rb') as f:
        reader = csv.reader(f)
        reader.next()
        for row in reader:
            clip = row[0]
            prediction = np.float64(row[1])
            clip2prediction[clip] = prediction

    clip2label, clip2time, clip2usage = loader.load_test_labels(pathfinder.LABELS_PATH)
    test_preictal_groups, test_interictal_groups = loader.group_labels_by_hour(clip2label, clip2time, subject)
    test_groups = test_preictal_groups
    correct = 0
    for i, group in enumerate(test_groups):
        min_j = 7
        c = 0
        for j, clip in enumerate(group):
            if clip2prediction[clip] > threshold:
                c += 1
                min_j = j if j < min_j else min_j
        if c >= 3:
            correct += 1
            # print 'group', i
            # print 'min detection t', min_j

    print 'correct/ n_seizures', correct, len(test_groups)


def fp(submission_file, subject, threshold):
    clip2prediction = {}
    with open(submission_file, 'rb') as f:
        reader = csv.reader(f)
        reader.next()
        for row in reader:
            clip = row[0]
            prediction = np.float64(row[1])
            clip2prediction[clip] = prediction

    clip2label, clip2time, clip2usage = loader.load_test_labels(pathfinder.LABELS_PATH)
    test_preictal_groups, test_interictal_groups = loader.group_labels_by_hour(clip2label, clip2time, subject)
    test_groups = test_interictal_groups
    wrong = 0
    for i, group in enumerate(test_groups):
        wrong_idxs = []
        for j, clip in enumerate(group):
            if clip2prediction[clip] > threshold:
                wrong_idxs.append(j)
        if len(wrong_idxs) >= 3:
            wrong += 1
        if wrong_idxs:
            print i, wrong_idxs

    print 'wrong / n_seizures', wrong, len(test_groups)


def per_group(submission_path, subject, threshold):
    clip2prediction = {}
    with open(submission_path, 'rb') as f:
        reader = csv.reader(f)
        reader.next()
        for row in reader:
            clip = row[0]
            prediction = np.float64(row[1])
            clip2prediction[clip] = prediction

    clip2label, clip2time, clip2usage = loader.load_test_labels(pathfinder.LABELS_PATH)
    test_preictal_groups, test_interictal_groups = loader.group_labels_by_hour(clip2label, clip2time, subject)
    test_groups = test_preictal_groups + test_interictal_groups

    fn, fp, tp, tn = 0, 0, 0, 0
    n_preictal, n_interictal = 0, 0
    group_preds, group_labels = [], []
    for i, group in enumerate(test_groups):
        if len(group) != 6:
            print 'skipping group with length', len(group)
        elif len(group) == 6:
            group_prediction = np.mean([clip2prediction[clip] for clip in group])
            group_preds.append(group_prediction)
            label = clip2label[group[0]]
            group_labels.append(label)
            if label == 1:
                n_preictal += 1
            else:
                n_interictal += 1

            if label == 1 and group_prediction < threshold:
                fn += 1
            elif label == 0 and group_prediction >= threshold:
                fp += 1
            elif label == 1 and group_prediction >= threshold:
                tp += 1
            elif label == 0 and group_prediction < threshold:
                tn += 1
            else:
                raise ValueError()

    print '# preictal', n_preictal
    print '# interictal', n_interictal
    print 'TP', tp
    print 'TN', tn
    print 'FP', fp
    print 'FN', fn
    try:
        print 'group AUC', auc(np.array(group_labels), np.array(group_preds))
    except:
        pass


def evaluate_submission(submission_file):
    clip2label, _, clip2usage = loader.load_test_labels(pathfinder.LABELS_PATH)
    clip2prediction = {}
    with open(submission_file, 'rb') as f:
        reader = csv.reader(f)
        reader.next()
        for row in reader:
            clip = row[0]
            prediction = np.float64(row[1])
            clip2prediction[clip] = prediction

    usage = ['Public', 'Private']
    for u in usage:
        targets, predictions = [], []
        for k, v in clip2label.iteritems():
            if clip2usage[k] == u:
                targets.append(v)
                predictions.append(clip2prediction[k])

        targets, predictions = np.array(targets), np.array(predictions)
        print
        print u, 'AUC:', auc(targets, predictions)
        print u, 'Sensitivity at 75% specificity:', utils.get_tpr(targets, predictions)

    targets, predictions = [], []
    for k, v in clip2label.iteritems():
        targets.append(v)
        predictions.append(clip2prediction[k])
    targets, predictions = np.array(targets), np.array(predictions)
    print
    print 'all AUC:', auc(targets, predictions)
    print ' all Sensitivity at 75% specificity:', utils.get_tpr(targets, predictions)


def evaluate_submission_per_subject(submission_file, subject):
    clip2label, _, clip2usage = loader.load_test_labels(pathfinder.LABELS_PATH)
    clip2prediction = {}
    with open(submission_file, 'rb') as f:
        reader = csv.reader(f)
        reader.next()
        for row in reader:
            clip = row[0]
            prediction = float(row[1])
            if subject in clip:
                clip2prediction[clip] = prediction

    targets, predictions = [], []
    for k, v in clip2label.iteritems():
        if subject in k:
            targets.append(v)
            predictions.append(clip2prediction[k])

    targets, predictions = np.array(targets), np.array(predictions)
    print subject, 'AUC:', auc(targets, predictions)
    print subject, 'Sensitivity at 75% specificity:', utils.get_tpr(targets, predictions)


def evaluate_submission_subjects(submission_file, subjects):
    clip2label, _, clip2usage = loader.load_test_labels(pathfinder.LABELS_PATH)
    clip2prediction = {}
    with open(submission_file, 'rb') as f:
        reader = csv.reader(f)
        reader.next()
        for row in reader:
            clip = row[0]
            prediction = np.float64(row[1])
            clip2prediction[clip] = prediction

    targets, predictions = [], []
    for subject in subjects:
        for k, v in clip2label.iteritems():
            if subject in k:
                targets.append(v)
                predictions.append(clip2prediction[k])

    targets, predictions = np.array(targets), np.array(predictions)
    print
    print 'AUC:', auc(targets, predictions)
    print 'Sensitivity at 75% specificity:', utils.get_tpr(targets, predictions)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: evaluate.py <config_name>")

    config_name = sys.argv[1]
    set_configuration(config_name)

    # submissions paths
    submission_dir = utils.get_dir_path('submissions', pathfinder.METADATA_PATH)
    submission_path = utils.find_model_metadata(submission_dir, config_name)
    subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    for subject in subjects:
        print '================================', subject
        evaluate_submission_per_subject(submission_path, subject)
        # tp_per_seizure(submission_path, subject, 0.5)
        # fp(submission_path, subject, 0.5)
        per_group(submission_path, subject, 0.5)
    print '==========================================='
    evaluate_submission(submission_path)
