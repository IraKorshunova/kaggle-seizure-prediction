import numpy as np
import json
from theano import config
import cPickle
from utils.loader import load_grouped_train_data, load_train_data, load_test_data
from utils.config_name_creator import *
from matplotlib import cm
import matplotlib.pyplot as plt

config.floatX = 'float32'


def plot_train_test_sizes(subjects, data_path):
    filenames_grouped_by_hour = cPickle.load(open('filenames.pickle'))
    dict_train, dict_test = {}, {}
    for subject in subjects:
        x_train = load_train_data(data_path, subject)['x']
        x_test = load_test_data(data_path, subject)['x']
        dict_train[subject] = x_train.shape[0]
        dict_test[subject] = x_test.shape[0]

    plt_data = [dict_train, dict_test]
    data_orders = [subjects, subjects]
    colors = [cm.Greys(1. * i / (len(subjects) + 2)) for i in range(len(subjects))]

    values = np.array([[data[name] for name in subjects] for data, order in zip(plt_data, data_orders)])
    lefts = np.insert(np.cumsum(values, axis=1), 0, 0, axis=1)[:, :-1]
    orders = np.array(data_orders)
    bottoms = np.array([0.1, 0.8])

    for name, color in zip(subjects, colors):
        idx = np.where(orders == name)
        value = values[idx]
        left = lefts[idx]
        plt.bar(left=left, height=0.3, width=value, bottom=bottoms,
                color=color, orientation="horizontal", label=name)
        left_margin0 = left[0] if value[0] < 100 else left[0] + value[0] / 3.0
        left_margin1 = left[1] if value[1] < 100 else left[1] + value[1] / 3.0
        plt.text(left_margin0, bottoms[0] + 0.15, value[0])
        plt.text(left_margin1, bottoms[1] + 0.15, value[1])
        if value[0] < 100:
             ratio = str(len(filenames_grouped_by_hour[name]['interictal'])) + '/\n' + str(len(filenames_grouped_by_hour[name]['preictal']))
        else:
            ratio = str(len(filenames_grouped_by_hour[name]['interictal'])) + '/' + str(len(filenames_grouped_by_hour[name]['preictal']))
        plt.text(left_margin0, bottoms[0]+0.08, ratio)

    plt.yticks(bottoms + 0.15, ["train", "test"])
    plt.legend(loc="best", bbox_to_anchor=(1.0, 1.00))
    plt.subplots_adjust(right=0.85)
    plt.show()


if __name__ == '__main__':
    with open('SETTINGS.json') as f:
        settings_dict = json.load(f)

    data_path = settings_dict['path']['processed_data_path'] + '/' + create_fft_data_name(settings_dict)
    subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    plot_train_test_sizes(subjects, data_path)
