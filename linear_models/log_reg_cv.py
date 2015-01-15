import numpy as np
import json, os
import preprocessors.fft as fft
from utils.loader import load_grouped_train_data, load_train_data
from utils.config_name_creator import *
from sklearn.linear_model import LogisticRegression
import cPickle
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_curve, auc
from commons import reshape_data


def predict(model, x_test, n_test_examples, n_timesteps):
    pred_1m = model.predict_proba(x_test)[:, 1]
    pred_10m = np.reshape(pred_1m, (n_test_examples, n_timesteps))
    pred_10m = np.mean(pred_10m, axis=1)
    return pred_10m


def cross_validate(subject, data_path, reg_C, random_cv=False):
    if random_cv:
        d = load_train_data(data_path,subject)
        x, y = d['x'], d['y']
        skf = StratifiedKFold(y, n_folds=10)
    else:
        filenames_grouped_by_hour = cPickle.load(open('filenames.pickle'))
        data_grouped_by_hour = load_grouped_train_data(data_path, subject, filenames_grouped_by_hour)
        n_preictal, n_interictal = len(data_grouped_by_hour['preictal']), len(data_grouped_by_hour['interictal'])
        hours_data = data_grouped_by_hour['preictal'] + data_grouped_by_hour['interictal']
        hours_labels = np.concatenate((np.ones(n_preictal), np.zeros(n_interictal)))
        n_folds = n_preictal
        skf = StratifiedKFold(hours_labels, n_folds=n_folds)


    preictal_probs, labels = [], []
    for train_indexes, valid_indexes in skf:
        x_train, x_valid = [], []
        y_train, y_valid = [], []
        for i in train_indexes:
            x_train.extend(hours_data[i])
            y_train.extend(hours_labels[i] * np.ones(len(hours_data[i])))
        for i in valid_indexes:
            x_valid.extend(hours_data[i])
            y_valid.extend(hours_labels[i] * np.ones(len(hours_data[i])))

        x_train = [x[..., np.newaxis] for x in x_train]
        x_train = np.concatenate(x_train, axis=3)
        x_train = np.rollaxis(x_train, axis=3)
        y_train = np.array(y_train)

        x_valid = [x[..., np.newaxis] for x in x_valid]
        x_valid = np.concatenate(x_valid, axis=3)
        x_valid = np.rollaxis(x_valid, axis=3)
        y_valid = np.array(y_valid)

        n_valid_examples = x_valid.shape[0]
        n_timesteps = x_valid.shape[-1]

        x_train, y_train = reshape_data(x_train, y_train)
        data_scaler = StandardScaler()
        x_train = data_scaler.fit_transform(x_train)

        logreg = LogisticRegression(C=reg_C)
        logreg.fit(x_train, y_train)

        x_valid = reshape_data(x_valid)
        x_valid = data_scaler.transform(x_valid)

        p_valid = predict(logreg, x_valid, n_valid_examples, n_timesteps)

        preictal_probs.extend(p_valid)
        labels.extend(y_valid)

    return preictal_probs, labels


def run_trainer():
    with open('SETTINGS.json') as f:
        settings_dict = json.load(f)

    # path
    data_path = settings_dict['path']['processed_data_path'] + '/' + create_fft_data_name(settings_dict)
    print data_path

    if not os.path.exists(data_path):
        fft.run_fft_preprocessor()

    subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    for reg_C in [10000000, 100, 10, 1.0, 0.1, 0.01]:
        print reg_C
        all_valid_probs = []
        all_valid_y = []
        for subject in subjects:
            p, y = cross_validate(subject, data_path, reg_C=reg_C)
            all_valid_probs.extend(p)
            all_valid_y.extend(y)

        fpr, tpr, _ = roc_curve(all_valid_y, all_valid_probs, pos_label=1)
        print auc(fpr, tpr)
        print log_loss(all_valid_y, all_valid_probs)


if __name__ == '__main__':
    run_trainer()