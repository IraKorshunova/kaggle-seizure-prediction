import numpy as np
import json
import cPickle

import matplotlib.pyplot as plt
from theano import config
import matplotlib.cm as cmx
import matplotlib.colors as colors
from sklearn.metrics import roc_curve

from utils.loader import load_train_data
from utils.config_name_creator import *
from utils.data_scaler import scale_across_features, scale_across_time
from cnn.conv_net import ConvNet


config.floatX = 'float32'


def get_cmap(N):
    color_norm = colors.Normalize(vmin=0, vmax=N - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')

    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)

    return map_index_to_rgb_color


def plot_train_probs(subject, data_path, model_path):
    with open(model_path + '/' + subject + '.pickle', 'rb') as f:
        state_dict = cPickle.load(f)
    cnn = ConvNet(state_dict['params'])
    cnn.set_weights(state_dict['weights'])
    scalers = state_dict['scalers']

    d = load_train_data(data_path, subject)
    x, y = d['x'], d['y']

    x, _ = scale_across_time(x, x_test=None, scalers=scalers) if state_dict['params']['scale_time'] \
        else scale_across_features(x, x_test=None, scalers=scalers)

    cnn.batch_size.set_value(x.shape[0])
    probs = cnn.get_test_proba(x)

    fpr, tpr, threshold = roc_curve(y, probs)
    c = np.sqrt((1-tpr)**2+fpr**2)
    opt_threshold = threshold[np.where(c==np.min(c))[0]]
    print opt_threshold

    x_coords = np.zeros(len(y), dtype='float64')
    rng = np.random.RandomState(42)
    x_coords += rng.normal(0.0, 0.08, size=len(x_coords))
    plt.scatter(x_coords, probs, c=y, s=60)
    plt.title(subject)
    plt.show()


if __name__ == '__main__':

    with open('SETTINGS.json') as f:
        settings_dict = json.load(f)
    data_path = settings_dict['path']['processed_data_path'] + '/' + create_fft_data_name(settings_dict)
    model_path = settings_dict['path']['model_path'] + '/' + create_cnn_model_name(settings_dict)
    subjects = ['Patient_1', 'Patient_2', 'Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5']
    for subject in subjects:
        print '***********************', subject, '***************************'
        plot_train_probs(subject, data_path, model_path)