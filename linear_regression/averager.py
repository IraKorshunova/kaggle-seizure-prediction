import numpy as np
import json, os, itertools, csv
from pandas import read_csv
import preprocessors.fft as fft
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame

def rescale(x):
    norm_x = StandardScaler().fit_transform(x)
    return 1.0 / (1.0 + np.exp(-norm_x))

def merge_csv_files(subjects, submission_name):
    print subjects
    with open(submission_name + '.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['clip', 'preictal'])

    for subject in subjects:
        df = read_csv(subject + '.csv')
        df['preictal'] = rescale(df.drop('clip', axis=1).values)
        with open(submission_name + '.csv', 'a') as f:
            df.to_csv(f, header=False, index=False)
        os.remove(subject +'.csv')


def average(subject, submission_paths):
    dfs = []
    for submission_path in submission_paths:
        dfs.append(read_csv(submission_path + '/' + subject + '.csv'))

    clips = dfs[0]['clip'].values

    avg = np.zeros(len(clips))
    for df in dfs:
        avg += df['preictal'].values
    avg /= len(dfs)

    ans = zip(clips, avg)
    df = DataFrame(data=ans, columns=['clip', 'preictal'])
    df.to_csv(subject + '.csv', index=False, header=True)



if __name__ =="__main__":
    with open('SETTINGS.json') as f:
        settings_dict = json.load(f)

    s1 = "fft_meanlog_lowcut0.1highcut180nfreq_bands67win_length_sec60stride_sec30"
    submission_path1 = settings_dict['path']['submission_path'] + '/' + s1

    s2 = "fft_meanlog_lowcut0.1highcut180nfreq_bands67win_length_sec60stride_sec30"
    submission_path2 = settings_dict['path']['submission_path'] + '/' + s2

    subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    for subject in subjects:
        print '***********************', subject, '***************************'
        average(subject,[submission_path1, submission_path2])
    merge_csv_files(subjects, 'submission_average')