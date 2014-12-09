import numpy as np
import json, os, itertools
import preprocessors.fft as fft
from theano import config
from pandas import DataFrame
from utils.loader import load_grouped_train_data, load_test_data, load_train_data
from utils.config_name_creator import *
from sklearn.preprocessing import StandardScaler
from utils.data_splitter import generate_overlapped_data
from sklearn.linear_model import RidgeCV, LinearRegression

config.floatX = 'float32'


def reshape_data(x, y=None):
    n_examples = x.shape[0]
    n_channels = x.shape[1]
    n_fbins = x.shape[2]
    n_timesteps = x.shape[3]
    x_new = np.zeros((n_examples * n_timesteps, n_channels, n_fbins))
    for i in range(n_channels):
        xi = np.transpose(x[:, i, :, :], axes=(0, 2, 1))
        xi = xi.reshape((n_examples * n_timesteps, n_fbins))
        x_new[:, i, :] = xi

    x_new = x_new.reshape((n_examples * n_timesteps, n_channels * n_fbins))
    if y is not None:
        y_new = np.repeat(y, n_timesteps)
        return x_new, y_new
    else:
        return x_new


def train(subject, data_path):
    d = load_train_data(data_path, subject)
    x, y = d['x'], d['y']
    n_channels = x.shape[1]
    n_fbins = x.shape[2]

    x, y = reshape_data(x, y)

    data_scaler = StandardScaler()
    x = data_scaler.fit_transform(x)

    ridge_reg = LinearRegression()
    ridge_reg.fit(x, y)

    #printing
    names = [x for x in itertools.product(range(n_channels), range(n_fbins))]
    zipped = zip(names, ridge_reg.coef_)
    print sorted(zipped, key=lambda x: np.abs(x[1]), reverse=True)
    return ridge_reg, data_scaler


def predict(subject, model, data_scaler, data_path, submission_path):
    d = load_test_data(data_path, subject)
    x_test, id = d['x'], d['id']
    n_test_examples = x_test.shape[0]
    n_timesteps = x_test.shape[3]

    x_test = reshape_data(x_test)
    x_test = data_scaler.transform(x_test)

    pred_1m = model.predict(x_test)
    pred_10m = np.reshape(pred_1m, (n_test_examples, n_timesteps))
    pred_10m = np.mean(pred_10m, axis=1)

    ans = zip(id, pred_10m)
    df = DataFrame(data=ans, columns=['clip', 'preictal'])
    df.to_csv(submission_path + '/' + subject + '.csv', index=False, header=True)


def run_trainer():
    with open('SETTINGS.json') as f:
        settings_dict = json.load(f)

    # path
    data_path = settings_dict['path']['processed_data_path'] + '/' + create_fft_data_name(settings_dict)
    submission_path = settings_dict['path']['submission_path'] + '/' + create_fft_data_name(settings_dict)
    print data_path

    if not os.path.exists(data_path):
        fft.run_fft_preprocessor()

    if not os.path.exists(submission_path):
        os.makedirs(submission_path)

    subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    for subject in subjects:
        print '***********************', subject, '***************************'
        model, data_scaler = train(subject, data_path)
        predict(subject, model, data_scaler, data_path, submission_path)


if __name__ == '__main__':
    run_trainer()