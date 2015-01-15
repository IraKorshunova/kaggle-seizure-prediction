import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas import DataFrame
from collections import defaultdict


def print_cm(cm, labels):
    columnwidth = max([len(x) for x in labels])
    # Print header
    print " " * columnwidth,
    for label in labels:
        print "%{0}s".format(columnwidth) % label,
    print
    # Print rows
    for i, label1 in enumerate(labels):
        print "%{0}s".format(columnwidth) % label1,
        for j in range(len(labels)):
            print "%{0}d".format(columnwidth) % cm[i, j],
        print


def load_test_labels(csv_path):
    subject_to_df = defaultdict(list)
    d = DataFrame.from_csv(csv_path, index_col=None)
    for i in d.index:
        clip = d['clip'][i]
        preictal = d['preictal'][i]

        subject_name = '_'.join(clip.split('_', 2)[:2])
        subject_to_df[subject_name].append((clip, preictal))

    for subject_name, subject_data in subject_to_df.iteritems():
        subject_to_df[subject_name] = DataFrame(subject_data, columns=['clip', 'preictal'])
    return subject_to_df


def softmax_scaler(x):
    norm_x = StandardScaler().fit_transform(x)
    return 1.0 / (1.0 + np.exp(-norm_x))


def minmax_scaler(x):
    scaler = MinMaxScaler(feature_range=(0.000000001, 0.999999999))
    return scaler.fit_transform(x)


def median_scaler(x):
    return (x - np.median(x)) / 2.0 + 0.5


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

