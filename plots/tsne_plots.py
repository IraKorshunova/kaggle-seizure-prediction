import matplotlib

matplotlib.use('Agg')
import numpy as np
import sys
import preprocess
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import loader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.cm as cmx
import matplotlib.colors as colors
from configuration import set_configuration, config
import pathfinder
import utils

marker_type = {u'D': u'diamond', u's': u'square', u'^': u'triangle_up', u'd': u'thin_diamond', u'h': u'hexagon1',
               u'*': u'star', u'o': u'circle', u'.': u'point', u'p': u'pentagon', u'H': u'hexagon2',
               u'v': u'triangle_down', u'8': u'octagon', u'<': u'triangle_left', u'>': u'triangle_right'}


def get_cmap(N):
    color_norm = colors.Normalize(vmin=0, vmax=N - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')

    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)

    return map_index_to_rgb_color


def tsne_groups_plot(subject):
    print 'Plotting groups'
    data_path = preprocess.preprocess_data(pathfinder.RAW_DATA_PATH, subject, config().transformation_params)

    train_groups_x, train_groups_y, _ = loader.load_grouped_train_data(data_path, subject)

    test_clip2label, test_clip2time, test_clip2usage = loader.load_test_labels(pathfinder.LABELS_PATH)
    test_preictal_groups, test_interictal_groups = loader.group_labels_by_hour(test_clip2label, test_clip2time, subject)
    test_grouped_filenames = test_preictal_groups + test_interictal_groups
    test_group_labels = [1] * len(test_preictal_groups) + [0] * len(test_interictal_groups)

    text, colors, markers = [], [], []

    # TRAIN
    x_train = []
    i = 0
    for hour, hour_labels in zip(train_groups_x, train_groups_y):
        assert len(set(hour_labels)) == 1

        for j, clip in enumerate(hour):
            x_train.append(np.reshape(clip, (1, clip.shape[0] * clip.shape[1] * clip.shape[2])))
            if hour_labels[0] == 1:
                text.append(str(i) + '.' + str(j))
            else:
                text.append(u' ')

        if hour_labels[0] == 1:
            i += 1

        if hour_labels[0] == 1:
            colors.extend(['red'] * len(hour))
            markers.extend([u'^'] * len(hour))
        else:
            colors.extend(['blue'] * len(hour))
            markers.extend([u'.'] * len(hour))

    x_train = np.vstack(x_train)
    print x_train.shape

    # TEST
    x_test = []
    for i, group in enumerate(test_grouped_filenames):
        for j, filename in enumerate(group):
            x, _, _ = loader.read_file(data_path + '/' + subject + '/' + filename, expand_dims=False)
            x_test.append(np.reshape(x, (1, x.shape[0] * x.shape[1] * x.shape[2])))
            l = test_group_labels[i]
            if l == 1:
                markers.append(u'v')
                colors.append('darkorange')
            else:
                markers.append(u'.')
                colors.append('skyblue')
            text.append('')

    x_test = np.vstack(x_test)
    print 'test', x_test.shape

    # ALL
    x_all = x_train  # np.vstack((np.float32(x_train), np.float32(x_test)))
    scaler = StandardScaler()
    x_all = scaler.fit_transform(x_all)
    print 'ALL shape', x_all.shape

    # TSNE
    pca = PCA(50)
    pca.fit(x_all)
    x_all = pca.fit_transform(x_all)

    tsne = TSNE(random_state=42)
    z = tsne.fit_transform(x_all)

    # PLOT
    plt.figure()
    print 'zip length', len(zip(z[:, 0], z[:, 1], colors, markers, text))

    for a, b, c, d, t in zip(z[:, 0], z[:, 1], colors, markers, text):
        plt.scatter(a, b, c=c, s=30, marker=d, linewidths=0.2)
        plt.annotate(t, (a, b))

    # zz = z[np.where(np.array(markers) != u' ')[0], :]
    ax = plt.subplot(111)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
              ncol=2, fancybox=True, shadow=True)
    plt.xlim([min(z[:, 0]) - 0.5, max(z[:, 0] + 0.5)])
    plt.ylim([min(z[:, 1]) - 0.5, max(z[:, 1] + 0.5)])
    plt.savefig(img_path + '/tsne-%s-%s-groups.png' % (subject, config_name))


def tsne_train_test_errors(subject, predictions_exp_dir):
    data_path = preprocess.preprocess_data(pathfinder.RAW_DATA_PATH, subject, config().transformation_params)

    x_train, y_train, filename2idx_train, idx2filename_train = loader.load_train_data(data_path, subject)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3])

    x_test, filename2idx_test, idx2filename_test = loader.load_test_data(data_path, subject)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3]))
    test_clip2label, test_clip2time, test_clip2usage = loader.load_test_labels(pathfinder.LABELS_PATH)

    markers, preds = [], []

    # TRAIN
    train_id2pred = utils.load_pkl(predictions_exp_dir + '/%s-train.pkl' % subject)
    train_id2label = loader.load_train_labels(pathfinder.LABELS_PATH)
    for i in xrange(len(x_train)):
        filename = idx2filename_train[i]
        preds.append(train_id2pred[filename])
        if train_id2label[filename.split('/')[-1]]:
            markers.append('^')
        else:
            markers.append('.')

    # TEST
    test_id2pred = utils.load_pkl(predictions_exp_dir + '/%s-test.pkl' % subject)
    for i in xrange(len(x_test)):
        filename = idx2filename_test[i]
        preds.append(test_id2pred[filename])
        if test_clip2label[filename]:
            markers.append('v')
        else:
            markers.append('.')

    # ALL
    x_all = np.vstack((np.float32(x_train), np.float32(x_test)))
    scaler = StandardScaler()
    x_all = scaler.fit_transform(x_all)
    print 'ALL shape', x_all.shape
    print len(markers)
    print len(preds)

    # TSNE
    pca = PCA(50)
    pca.fit(x_all)
    x_all = pca.fit_transform(x_all)

    tsne = TSNE(random_state=42)
    z = tsne.fit_transform(x_all)

    # PLOT
    plt.figure()
    zipped = zip(z[:, 0], z[:, 1], markers, preds)
    zipped.sort(key=lambda t: t[3])
    for a, b, d, e in zipped:
        plt.scatter(a, b, c=e, s=30, marker=d, linewidths=0.2, edgecolor='black', cmap="Reds", vmin=0., vmax=1.)

    # plt.scatter(z[0, 0], z[0, 1], c=errors[0], marker=markers[0], s=40, edgecolor='black', linewidths=0.2,
    #             label='train',
    #             cmap="Reds", vmin=0., vmax=1.)
    # plt.scatter(z[-1, 0], z[-1, 1], c=errors[-1], marker=markers[-1], s=40, edgecolor='black', linewidths=0.2,
    #             label='test', cmap="Reds", vmin=0., vmax=1.)

    plt.colorbar()

    ax = plt.subplot(111)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=2, fancybox=True, shadow=True)
    plt.xlim([min(z[:, 0]) - 0.5, max(z[:, 0] + 0.5)])
    plt.ylim([min(z[:, 1]) - 0.5, max(z[:, 1] + 0.5)])
    plt.savefig(img_path + '/tsne-%s-%s-errors.png' % (subject, config_name))


def tsne_train_test_plot(subject):
    print 'Plotting test train'
    data_path = preprocess.preprocess_data(pathfinder.RAW_DATA_PATH, subject, config().transformation_params)

    x_train, y_train, filename2idx_train, idx2filename_train = loader.load_train_data(data_path, subject)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3])
    train_clip2label = loader.load_train_labels(pathfinder.LABELS_PATH)

    x_test, filename2idx_test, idx2filename_test = loader.load_test_data(data_path, subject)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3]))
    test_clip2label, test_clip2time, test_clip2usage = loader.load_test_labels(pathfinder.LABELS_PATH)

    x_all = np.vstack((np.float32(x_train), np.float32(x_test)))
    print 'All schape', x_all.shape

    scaler = StandardScaler()
    x_all = scaler.fit_transform(x_all)

    colors, markers = [], []
    for i in xrange(len(x_train)):
        filename = idx2filename_train[i]
        label = train_clip2label[filename]
        if label == 1:
            colors.append('r')
            markers.append('^')
        else:
            colors.append('b')
            markers.append('.')

    for i in xrange(len(x_test)):
        filename = idx2filename_test[i]
        label = test_clip2label[filename]
        if label == 1:
            colors.append('salmon')
            markers.append('d')
        else:
            colors.append('skyblue')
            markers.append('*')

    pca = PCA(50)
    pca.fit(x_all)
    x_all = pca.fit_transform(x_all)

    tsne = TSNE(random_state=42)
    z = tsne.fit_transform(x_all)

    plt.figure()
    print 'zip length', len(zip(z[:, 0], z[:, 1], colors, markers))
    for a, b, c, d in zip(z[:, 0], z[:, 1], colors, markers):
        plt.scatter(a, b, c=c, s=40, marker=d)

    plt.scatter(z[0, 0], z[0, 1], c=colors[0], marker=markers[0], s=40, label='train')
    plt.scatter(z[-1, 0], z[-1, 1], c=colors[-1], marker=markers[-1], s=40, label='test')

    ax = plt.subplot(111)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=2, fancybox=True, shadow=True)
    plt.xlim([min(z[:, 0]) - 1, max(z[:, 0] + 1)])
    plt.ylim([min(z[:, 1]) - 1, max(z[:, 1] + 1)])
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)
    # plt.ylabel('Z_2', fontsize=20)
    # plt.xlabel('Z_1', fontsize=20)
    plt.savefig(img_path + '/tsne-%s-%s-test-train.png' % (subject, config_name))


def tsne_train_test_holdout_plot(subject):
    data_path = preprocess.preprocess_data(pathfinder.RAW_DATA_PATH, subject, config().transformation_params)

    x_train, y_train, filename2idx_train, idx2filename_train = loader.load_train_data(data_path, subject)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3])
    train_clip2label = loader.load_train_labels(pathfinder.LABELS_PATH)

    x_test, filename2idx_test, idx2filename_test = loader.load_test_data(data_path, subject)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3]))
    test_clip2label, test_clip2time, test_clip2usage = loader.load_test_labels(pathfinder.LABELS_PATH)

    subject_holdout = subject + '_holdout'
    x_holdout, filename2idx_holdout, idx2filename_holdout = loader.load_holdout_data(data_path, subject_holdout)
    x_holdout = np.reshape(x_holdout,
                           (x_holdout.shape[0], x_holdout.shape[1] * x_holdout.shape[2] * x_holdout.shape[3]))

    holdout_clip2label, _ = loader.load_holdout_labels(pathfinder.LABELS_PATH)

    x_all = np.vstack((np.float32(x_train), np.float32(x_test), np.float32(x_holdout)))
    print x_all.shape

    scaler = StandardScaler()
    x_all = scaler.fit_transform(x_all)

    colors = []
    for i in xrange(len(x_train)):
        filename = idx2filename_train[i]
        label = train_clip2label[filename]
        if label == 1:
            colors.append('r')
        else:
            colors.append('r')

    for i in xrange(len(x_test)):
        filename = idx2filename_test[i]
        label = test_clip2label[filename]
        if label == 1:
            colors.append('b')
        else:
            colors.append('b')

    for i in xrange(len(x_holdout)):
        filename = idx2filename_holdout[i]
        label = holdout_clip2label[filename]
        if label == 1:
            colors.append('m')
        else:
            colors.append('m')

    markers = ['o'] * len(x_train) + ['^'] * len(x_test) + ['v'] * len(x_holdout)

    pca = PCA(50)
    pca.fit(x_all)
    x_all = pca.fit_transform(x_all)

    tsne = TSNE(random_state=42)
    z = tsne.fit_transform(x_all)

    plt.figure()
    for a, b, c, d in zip(z[:, 0], z[:, 1], colors, markers):
        plt.scatter(a, b, c=c, s=40, marker=d)

    plt.scatter(z[0, 0], z[0, 1], c=colors[0], marker=markers[0], s=40, label='train')
    plt.scatter(z[-1, 0], z[-1, 1], c=colors[-1], marker=markers[-1], s=40, label='test')

    ax = plt.subplot(111)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=2, fancybox=True, shadow=True)
    plt.xlim([min(z[:, 0]) - 1, max(z[:, 0] + 1)])
    plt.ylim([min(z[:, 1]) - 1, max(z[:, 1] + 1)])
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)
    plt.ylabel('Z_2', fontsize=20)
    plt.xlabel('Z_1', fontsize=20)
    plt.savefig(img_path + '/tsne-%s-%s-test-train.png' % (subject, config_name))



def tsne_test_holdout_plot(subject):
    data_path = preprocess.preprocess_data(pathfinder.RAW_DATA_PATH, subject, config().transformation_params)
    x_test, filename2idx_test, idx2filename_test = loader.load_test_data(data_path, subject)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3]))
    test_clip2label, test_clip2time, test_clip2usage = loader.load_test_labels(pathfinder.LABELS_PATH)

    subject_holdout = subject + '_holdout'
    x_holdout, filename2idx_holdout, idx2filename_holdout = loader.load_holdout_data(data_path, subject_holdout)
    x_holdout = np.reshape(x_holdout,
                           (x_holdout.shape[0], x_holdout.shape[1] * x_holdout.shape[2] * x_holdout.shape[3]))

    holdout_clip2label, _ = loader.load_holdout_labels(pathfinder.LABELS_PATH)

    x_all = np.vstack((np.float32(x_test), np.float32(x_holdout)))
    print x_all.shape

    scaler = StandardScaler()
    x_all = scaler.fit_transform(x_all)

    colors = []
    for i in xrange(len(x_test)):
        filename = idx2filename_test[i]
        label = test_clip2label[filename]
        if test_clip2usage[filename] == 'Private':
            colors.append('r')
        else:
            colors.append('y')


    for i in xrange(len(x_holdout)):
        filename = idx2filename_holdout[i]
        label = holdout_clip2label[filename]
        if label == 1:
            colors.append('b')
        else:
            colors.append('b')

    markers = ['o'] * len(x_test) + ['^'] * len(x_holdout)

    pca = PCA(50)
    pca.fit(x_all)
    x_all = pca.fit_transform(x_all)

    tsne = TSNE(random_state=42)
    z = tsne.fit_transform(x_all)

    plt.figure()
    for a, b, c, d in zip(z[:, 0], z[:, 1], colors, markers):
        plt.scatter(a, b, c=c, s=40, marker=d, alpha=0.5)

    plt.scatter(z[0, 0], z[0, 1], c=colors[0], marker=markers[0], s=40, label='train')
    plt.scatter(z[-1, 0], z[-1, 1], c=colors[-1], marker=markers[-1], s=40, label='test')

    ax = plt.subplot(111)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=2, fancybox=True, shadow=True)
    plt.xlim([min(z[:, 0]) - 1, max(z[:, 0] + 1)])
    plt.ylim([min(z[:, 1]) - 1, max(z[:, 1] + 1)])
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)
    plt.ylabel('Z_2', fontsize=20)
    plt.xlabel('Z_1', fontsize=20)
    plt.savefig(img_path + '/tsne-%s-%s-test-holdout.png' % (subject, config_name))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit("Usage: evaluate.py <config_name>")

    config_name = sys.argv[1]
    set_configuration(config_name)

    prediction_dir = utils.get_dir_path('predictions', pathfinder.METADATA_PATH)
    predictions_exp_dir = utils.find_model_metadata(prediction_dir, config_name)
    img_path = pathfinder.IMG_PATH

    subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4']
    for subject in subjects:
        print subject
        # tsne_train_test_errors(subject, predictions_exp_dir)
        # tsne_train_test_plot(subject)
        # tsne_groups_plot(subject)
        # tsne_train_test_holdout_plot(subject)
        tsne_test_holdout_plot(subject)
