import csv
import json, copy, os
from pandas import read_csv
import numpy as np
from config_name_creator import create_cnn_model_name
from pandas import DataFrame


def average(p1_list):
    n_models = len(p1_list)
    p1_list = [np.array(p1) for p1 in p1_list]
    p0_list = [1.0 - p1 for p1 in p1_list]
    p1_unnorm = np.ones_like(p1_list[0])
    p0_unnorm = np.ones_like(p0_list[0])
    for i in range(n_models):
        p1_unnorm *= p1_list[i]
        p0_unnorm *= p0_list[i]

    p1_unnorm = np.power(p1_unnorm, 1.0 / n_models)
    p0_unnorm = np.power(p0_unnorm, 1.0 / n_models)
    z = p1_unnorm + p0_unnorm
    return p1_unnorm / z


def average_submission_files(submission_names, out_submission_name):
    with open(out_submission_name + '.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['clip', 'preictal'])

    dfs = [read_csv(submission_name) for submission_name in submission_names]
    p1_list = [df['preictal'].values for df in dfs]
    p1_avg = average(p1_list).squeeze()
    id = dfs[0]['clip'].values
    df = DataFrame(data=zip(id, p1_avg), columns=['clip', 'preictal'])
    with open(out_submission_name + '.csv', 'a') as f:
        df.to_csv(f, header=False, index=False)


if __name__ == '__main__':

    ensemble0_settings = ['0.81448', '0.80888', '0.80614', '0.80413', '0.80192', '0.79964']

    ensemble1_settings = ['0.81448', '0.80888', '0.80872', '0.80614', '0.80533', '0.80413', '0.80192', '0.79964', '0.79886',
                          '0.79882', '0.79865', '0.79833', '0.78967', '0.78953', '0.78612']

    ensemble2_settings = ['0.81448', '0.80888', '0.80614', '0.80413', '0.80192', '0.79964', '0.79833', '0.79882']

    ensemble3_settings = ['0.81448', '0.80888', '0.80614', '0.80533', '0.80413', '0.80192']

    ensemble4_settings = ['0.81448', '0.80888', '0.80614', '0.80533', '0.80413', '0.80192', '0.79964']

    ensemble5_settings = ['0.81448', '0.80888', '0.80614', '0.80533', '0.80413', '0.80192', '0.79964', '0.79886',
                          '0.79882', '0.79865', '0.79833']

    ensemble6_settings = ['0.81448', '0.80888', '0.80872', '0.80614', '0.80533', '0.80413', '0.80192', '0.79964']

    ensemble7_settings = ['0.81448', '0.80888', '0.80872', '0.80614', '0.80533', '0.80413', '0.80192']

    ensemble8_settings = ['0.81448', '0.80888', '0.80872', '0.80614', '0.80533']

    # final model: 0.78513 on private LB
    ensemble9_settings = ['0.80872', '0.80614', '0.80533', '0.80192', '0.79964', '0.79886', '0.79882', '0.79865',
                          '0.79833', '0.78967', '0.78612']


    ensembles_settings = [ensemble0_settings, ensemble1_settings, ensemble2_settings, ensemble3_settings,
                          ensemble4_settings, ensemble5_settings, ensemble6_settings, ensemble7_settings,
                          ensemble8_settings, ensemble9_settings]

    for i, ensemble_setting in enumerate(ensembles_settings):
        ensemble_setting = ['settings_dir/SETTINGS_' + s + '.json' for s in ensemble_setting]
        submissions = []
        for settings_file in ensemble_setting:
            with open(settings_file) as f:
                settings_dict = json.load(f)
            submission_path = settings_dict['path']['submission_path'] + '/' + create_cnn_model_name(settings_dict)
            submission_path += '/submission_scaled.csv'
            submissions.append(submission_path)
        average_submission_files(submissions, 'submission_average' + str(i))

