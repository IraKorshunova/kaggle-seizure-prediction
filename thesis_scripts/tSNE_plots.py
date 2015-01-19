import numpy as np
import json
import theano
import cPickle
from sklearn.manifold import TSNE
from theano import Param
import matplotlib.pyplot as plt
from theano import config
from utils.loader import load_grouped_train_data, load_test_data, load_train_data
from utils.config_name_creator import *
from utils.data_scaler import scale_across_features, scale_across_time
from cnn.conv_net import ConvNet
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.cm as cmx
import matplotlib.colors as colors
from test_labels_loader import load_test_labels

config.floatX = 'float32'


def get_cmap(N):
    color_norm = colors.Normalize(vmin=0, vmax=N - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')

    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)

    return map_index_to_rgb_color


def plot_features(subject, data_path, model_path, test_labels, dataset='test'):
    with open(model_path + '/' + subject + '.pickle', 'rb') as f:
        state_dict = cPickle.load(f)
    cnn = ConvNet(state_dict['params'])
    cnn.set_weights(state_dict['weights'])
    scalers = state_dict['scalers']

    if dataset == 'test':
        d = load_test_data(data_path, subject)
        x = d['x']
        y = test_labels['preictal']
    elif dataset == 'train':
        d = load_train_data(data_path, subject)
        x, y = d['x'], d['y']
    else:
        raise ValueError('dataset')

    x, _ = scale_across_time(x, x_test=None, scalers=scalers) if state_dict['params']['scale_time'] \
        else scale_across_features(x, x_test=None, scalers=scalers)

    cnn.batch_size.set_value(x.shape[0])
    get_features = theano.function([cnn.x, Param(cnn.training_mode, default=0)], cnn.feature_extractor.output,
                                 allow_input_downcast=True)

    logits_test = get_features(x)
    model = TSNE(n_components=2, random_state=0)
    z = model.fit_transform(np.float64(logits_test))
    plt.scatter(z[:, 0], z[:, 1], s=60, c=y)
    plt.show()


def plot_train_test(subject, data_path):
    d = load_train_data(data_path, subject)
    x_train = d['x']
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3])

    d = load_test_data(data_path, subject)
    x_test, id = d['x'], d['id']
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3]))

    x_all = np.vstack((np.float64(x_train), np.float64(x_test)))
    scaler = StandardScaler()
    x_all = scaler.fit_transform(x_all)

    colors = ['r'] * len(x_train) + ['b'] * len(x_test)
    markers = ['o'] * len(x_train) + ['^'] * len(x_test)

    pca = PCA(50)
    pca.fit(x_all)
    x_all = pca.fit_transform(x_all)

    model = TSNE(n_components=2, perplexity=40, learning_rate=100, random_state=42)
    z = model.fit_transform(x_all)

    for a, b, c, d in zip(z[:, 0], z[:, 1], colors, markers):
        plt.scatter(a, b, c=c, s=60, marker=d)

    plt.scatter(z[0, 0], z[0, 1], c=colors[0], marker=markers[0], s=60, label='train')
    plt.scatter(z[-1, 0], z[-1, 1], c=colors[-1], marker=markers[-1], s=60, label='test')

    zz = z[np.where(np.array(markers) != u' ')[0], :]
    ax = plt.subplot(111)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=2, fancybox=True, shadow=True)
    plt.xlim([min(zz[:, 0]) - 0.5, max(zz[:, 0] + 0.5)])
    plt.ylim([min(zz[:, 1]) - 0.5, max(zz[:, 1] + 0.5)])
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)
    plt.ylabel('Z_2', fontsize=20)
    plt.xlabel('Z_1', fontsize=20)
    plt.show()


def plot_sequences(subject, data_path, test_labels):
    # data train
    filenames_grouped_by_hour = cPickle.load(open('filenames.pickle'))
    data_grouped_by_hour = load_grouped_train_data(data_path, subject, filenames_grouped_by_hour)

    interictal_hours = data_grouped_by_hour['interictal']
    preictal_hours = data_grouped_by_hour['preictal']

    marker_type = {u'D': u'diamond', u's': u'square', u'^': u'triangle_up', u'd': u'thin_diamond', u'h': u'hexagon1',
                   u'*': u'star', u'o': u'circle', u'.': u'point', u'p': u'pentagon', u'H': u'hexagon2',
                   u'v': u'triangle_down', u'8': u'octagon', u'<': u'triangle_left', u'>': u'triangle_right'}
    marker_list = marker_type.keys() * 50

    x_train, colors, markers = [], [], []
    cmap = get_cmap(len(preictal_hours))

    print len(preictal_hours)
    for i, hour in enumerate(preictal_hours):
        for clip in hour:
            x_train.append(np.reshape(clip, (1, clip.shape[0] * clip.shape[1] * clip.shape[2])))
        colors.extend([cmap(i)] * len(hour))
        markers.extend([marker_list[i]] * len(hour))

    for i, hour in enumerate(interictal_hours):
        for clip in hour:
            x_train.append(np.reshape(clip, (1, clip.shape[0] * clip.shape[1] * clip.shape[2])))
        colors.extend(['r'] * len(hour))
        markers.extend([u' '] * len(hour))

    x_train = np.vstack(x_train)
    print x_train.shape

    d = load_test_data(data_path, subject)
    x_test, id = d['x'], d['id']
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3]))
    color_test = test_labels['preictal']
    print np.sum(test_labels['preictal'])
    color_test[np.where(color_test == 0)[0]] = 'b'
    color_test[np.where(color_test == 1)[0]] = 'b'

    colors.extend(list(color_test))
    markers.extend([u' '] * len(x_test))

    x_all = np.vstack((np.float64(x_train), np.float64(x_test)))
    scaler = StandardScaler()
    x_all = scaler.fit_transform(x_all)

    pca = PCA(50)
    pca.fit(x_all)
    x_all = pca.fit_transform(x_all)

    model = TSNE(n_components=2, perplexity=40, learning_rate=100, random_state=42)
    z = model.fit_transform(x_all)
    prev_c, i = 0, 0
    for a, b, c, d in zip(z[:, 0], z[:, 1], colors, markers):
        if c != prev_c and d != u' ':
            plt.scatter(a, b, c=c, s=70, marker=d, label=str(i))
            i += 1
        else:
            plt.scatter(a, b, c=c, s=70, marker=d)
        prev_c = c

    zz = z[np.where(np.array(markers) != u' ')[0], :]
    ax = plt.subplot(111)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=2, fancybox=True, shadow=True)
    plt.xlim([min(zz[:, 0]) - 0.5, max(zz[:, 0] + 0.5)])
    plt.ylim([min(zz[:, 1]) - 0.5, max(zz[:, 1] + 0.5)])
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)
    plt.ylabel('Z_2', fontsize=20)
    plt.xlabel('Z_1', fontsize=20)
    plt.show()


if __name__ == '__main__':

    with open('SETTINGS.json') as f:
        settings_dict = json.load(f)
    data_path = settings_dict['path']['processed_data_path'] + '/' + create_fft_data_name(settings_dict)
    model_path = settings_dict['path']['model_path'] + '/' + create_cnn_model_name(settings_dict)

    test_labels_path = '/mnt/sda4/CODING/python/kaggle_data/test_labels.csv'
    labels_df = load_test_labels(test_labels_path)

    subjects = ['Patient_1', 'Patient_2', 'Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5']
    for subject in subjects:
        print '***********************', subject, '***************************'
        plot_train_test(subject, data_path)
        plot_features(subject, data_path, model_path, labels_df[subject], dataset='train')
        plot_sequences(subject, data_path, labels_df[subject])