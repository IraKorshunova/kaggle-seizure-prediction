import csv
import json
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from utils.config_name_creator import create_fft_data_name
import numpy as np


def rescale(x):
    norm_x = StandardScaler().fit_transform(x)
    return 1.0 / (1.0 + np.exp(-norm_x))


def merge_csv_files(submission_path, subjects, submission_name):
    print subjects
    with open(submission_path + '/' + submission_name + '.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['clip', 'preictal'])

    for subject in subjects:
        df = read_csv(submission_path + '/' + subject + '.csv')
        df['preictal'] = rescale(df.drop('clip', axis=1).values)
        with open(submission_path + '/' + submission_name + '.csv', 'a') as f:
            df.to_csv(f, header=False, index=False)


if __name__ == '__main__':
    with open('SETTINGS.json') as f:
        settings_dict = json.load(f)

    submission_path = submission_path = settings_dict['path']['submission_path'] +'/'+ create_fft_data_name(settings_dict)
    print submission_path

    subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    merge_csv_files(submission_path, subjects, 'submission')