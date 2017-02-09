import utils
import csv
import numpy as np
from kaggle_auc import auc


def load_labels():
    csv_filepath = '/mnt/sda3/data/kaggle-seizure-prediction/labels/answer.csv'#'holdout_key.csv'
    clip2label = {}
    with open(csv_filepath, 'rb') as f:
        reader = csv.reader(f)
        reader.next()
        for row in reader:
            clip = row[0]
            label = int(row[1])
            clip2label[clip] = label
    return clip2label


def evaluate_submission(submission_file):
    clip2label = load_labels()
    clip2prediction = {}
    with open(submission_file, 'rb') as f:
        reader = csv.reader(f)
        reader.next()
        for row in reader:
            clip = row[0]
            prediction = np.float64(row[1])
            clip2prediction[clip] = prediction

    targets, predictions = [], []
    for k, v in clip2label.iteritems():
        targets.append(v)
        predictions.append(clip2prediction[k])

    targets, predictions = np.array(targets), np.array(predictions)
    print
    print 'AUC:', auc(targets, predictions)
    print 'Sensitivity at 75% specificity:', utils.get_tpr(targets, predictions)


def evaluate_submission_per_subject(submission_file, subject):
    clip2label = load_labels()
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


if __name__ == "__main__":
    submission_path = 'submission_average0.csv'
    evaluate_submission(submission_path)
    # print '==========================================='
    # subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    # for subject in subjects:
    #     print '================================', subject
    #     evaluate_submission_per_subject(submission_path, subject)
