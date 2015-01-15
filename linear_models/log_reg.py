import numpy as np
import json, os
import preprocessors.fft as fft
from pandas import DataFrame
from utils.loader import load_test_data, load_train_data
from utils.config_name_creator import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from merger import merge_csv_files
from commons import reshape_data


def train(subject, data_path, reg_C=None):
    d = load_train_data(data_path, subject)
    x, y = d['x'], d['y']
    x, y = reshape_data(x, y)
    data_scaler = StandardScaler()
    x = data_scaler.fit_transform(x)
    lda = LogisticRegression(C=reg_C)
    lda.fit(x, y)
    return lda, data_scaler


def predict(subject, model, data_scaler, data_path, submission_path):
    d = load_test_data(data_path, subject)
    x_test, id = d['x'], d['id']
    n_test_examples = x_test.shape[0]
    n_timesteps = x_test.shape[3]

    x_test = reshape_data(x_test)
    x_test = data_scaler.transform(x_test)

    pred_1m = model.predict_proba(x_test)[:, 1]

    pred_10m = np.reshape(pred_1m, (n_test_examples, n_timesteps))
    pred_10m = np.mean(pred_10m, axis=1)

    ans = zip(id, pred_10m)
    df = DataFrame(data=ans, columns=['clip', 'preictal'])
    df.to_csv(submission_path + '/' + subject + '.csv', index=False, header=True)
    return pred_10m


def run_trainer():
    with open('SETTINGS.json') as f:
        settings_dict = json.load(f)

    reg_list = [10000000, 100, 10, 1.0, 0.1, 0.01]
    for reg_C in reg_list:
        print reg_C
        data_path = settings_dict['path']['processed_data_path'] + '/' + create_fft_data_name(settings_dict)
        submission_path = settings_dict['path']['submission_path'] + '/logreg_' + str(
            reg_C) + '_' + create_fft_data_name(settings_dict)

        if not os.path.exists(data_path):
            fft.run_fft_preprocessor()

        if not os.path.exists(submission_path):
            os.makedirs(submission_path)

        subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
        for subject in subjects:
            print subject
            model, data_scaler, = train(subject, data_path, reg_C)
            predict(subject, model, data_scaler, data_path, submission_path)

        merge_csv_files(submission_path, subjects, 'submission')
        merge_csv_files(submission_path, subjects, 'submission_softmax')
        merge_csv_files(submission_path, subjects, 'submission_minmax')
        merge_csv_files(submission_path, subjects, 'submission_median')


if __name__ == '__main__':
    run_trainer()