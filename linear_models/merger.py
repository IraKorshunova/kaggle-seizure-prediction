import csv
import json
from pandas import read_csv
from utils.config_name_creator import create_fft_data_name
from commons import softmax_scaler,minmax_scaler, median_scaler


def merge_csv_files(submission_path, subjects, submission_name):
    print subjects
    with open(submission_path + '/' + submission_name + '.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['clip', 'preictal'])

    for subject in subjects:
        df = read_csv(submission_path + '/' + subject + '.csv')
        if 'softmax' in submission_name:
            df['preictal'] = softmax_scaler(df.drop('clip', axis=1).values)
        elif 'minmax' in submission_name:
            df['preictal'] = minmax_scaler(df.drop('clip', axis=1).values)
        elif 'median' in submission_name:
            df['preictal'] = median_scaler(df.drop('clip', axis=1).values)
        else:
            df['preictal'] = df.drop('clip', axis=1).values
        with open(submission_path + '/' + submission_name + '.csv', 'a') as f:
            df.to_csv(f, header=False, index=False)


if __name__ == '__main__':
    with open('SETTINGS.json') as f:
        settings_dict = json.load(f)

    submission_path = submission_path = settings_dict['path']['submission_path'] + '/logreg' + create_fft_data_name(
        settings_dict)
    print submission_path

    subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    merge_csv_files(submission_path, subjects, 'submission')
    merge_csv_files(submission_path, subjects, 'submission_softmax')
    merge_csv_files(submission_path, subjects, 'submission_minmax')
    merge_csv_files(submission_path, subjects, 'submission_median')
