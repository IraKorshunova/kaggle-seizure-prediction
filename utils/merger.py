import csv
import json
from pandas import read_csv
from config_name_creator import *
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def rescale(probability):
    scaler = MinMaxScaler(feature_range=(0.000000001, 0.999999999))
    return scaler.fit_transform(probability)


def merge_csv_files(submission_path, subjects, submission_name, scale=True):
    submission_name += '_scaled' if scale else ''

    print subjects
    with open(submission_path + '/' + submission_name + '.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['clip', 'preictal'])

    for subject in subjects:
        df = read_csv(submission_path + '/' + subject + '.csv')
        print min(df['preictal'])
        if scale:
            df['preictal'] = rescale(df.drop('clip', axis=1).values)
            print min(df['preictal'])
        with open(submission_path + '/' + submission_name + '.csv', 'a') as f:
            df.to_csv(f, header=False, index=False)


if __name__ == '__main__':
    with open('../SETTINGS.json') as f:
        settings_dict = json.load(f)

    submission_path = settings_dict['path']['submission_path'] + '/' + create_cnn_model_name(settings_dict)
    print submission_path

    subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    merge_csv_files(submission_path, subjects, 'submission', scale=True)