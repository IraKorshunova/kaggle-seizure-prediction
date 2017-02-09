import os
import csv
import numpy as np
from scipy.io import loadmat
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import utils
import pathfinder
import glob


def load_holdout_labels(dir_path):
    csv_filepath = dir_path + '/holdout_key.csv'
    clip2label = {}
    clips_csv, labels_csv = [], []
    with open(csv_filepath, 'rb') as f:
        reader = csv.reader(f)
        reader.next()
        for row in reader:
            clip = row[0]
            label = int(row[1])
            clip2label[clip] = label
            clips_csv.append(clip)
            labels_csv.append(label)

    clip2time = {}
    subjects = ['Dog_1_holdout', 'Dog_2_holdout', 'Dog_3_holdout', 'Dog_4_holdout']
    for s in subjects:
        d = loadmat(dir_path + '/%s_answer_key.mat' % s)
        times_mat = d['answer_key']['testing_uutc'][0][0][0] * 2.77778e-10
        labels_mat = d['answer_key']['classification_vector'][0][0][0]

        filtered_labels_csv, filtered_clips_csv = [], []
        for i in xrange(len(clips_csv)):
            if s in clips_csv[i]:
                filtered_clips_csv.append(clips_csv[i])
                filtered_labels_csv.append(labels_csv[i])

        tmp_labels = [-1 if l == 0 else 1 for l in filtered_labels_csv]
        assert np.equal(tmp_labels, labels_mat).all()

        for l_mat, l_csv, c_csv, t_mat in zip(labels_mat, filtered_labels_csv, filtered_clips_csv, times_mat):
            l = 0 if l_mat == -1 else 1
            assert l == l_csv
            clip2time[c_csv] = t_mat

    return clip2label, clip2time


def load_test_labels(dir_path):
    csv_filepath = dir_path + '/answer.csv'
    clip2label, clip2usage = {}, {}
    clips_csv, labels_csv = [], []
    with open(csv_filepath, 'rb') as f:
        reader = csv.reader(f)
        reader.next()
        for row in reader:
            clip = row[0]
            label = int(row[1])
            usage = row[2]
            clip2label[clip] = label
            clip2usage[clip] = usage
            clips_csv.append(clip)
            labels_csv.append(label)

    d1 = loadmat(dir_path + '/Dog_1_answer_key.mat')
    d2 = loadmat(dir_path + '/Dog_2_answer_key.mat')
    d3 = loadmat(dir_path + '/Dog_3_answer_key.mat')
    d4 = loadmat(dir_path + '/Dog_4_answer_key.mat')
    d5 = loadmat(dir_path + '/Dog_5_answer_key.mat')
    d6 = loadmat(dir_path + '/Patient_1_answer_key.mat')
    d7 = loadmat(dir_path + '/Patient_2_answer_key.mat')

    labels_mat, times_mat = [], []
    for d in [d1, d2, d3, d4, d5, d6, d7]:
        labels_mat.extend(d['answer_key'][0][0][1][0])
        times_mat.extend(d['answer_key'][0][0][3][0])

    clip2time = {}
    for l_mat, l_csv, c_csv, t_mat in zip(labels_mat, labels_csv, clips_csv, times_mat):
        l = 0 if l_mat == -1 else 1
        assert l == l_csv
        clip2time[c_csv] = t_mat * 2.77778e-10  # convert to hours

    return clip2label, clip2time, clip2usage


def load_train_labels(dir_path):
    clip2label = {}
    csv_filepath = dir_path + '/answer-train.csv'
    if os.path.isfile(csv_filepath):
        with open(csv_filepath, 'rb') as f:
            reader = csv.reader(f)
            reader.next()
            for row in reader:
                clip = row[0]
                label = int(row[1])
                clip2label[clip] = label
    else:
        all_filenames = glob.glob(pathfinder.RAW_DATA_PATH + '/*/*.mat')
        filenames = [os.path.basename(f) for f in all_filenames if 'test' not in f]
        with open(csv_filepath, 'wb') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['clip', 'preictal'])
            for f in filenames:
                clip2label[f] = 1 if 'preictal' in f else 0
                csv_writer.writerow([f, clip2label[f]])

    return clip2label


def group_labels_by_hour(clip2label, clip2time, subject):
    s_clip2label = utils.get_subject_records(clip2label, subject)
    s_clip2time = utils.get_subject_records(clip2time, subject)

    preictals, interictals = [], []
    for k in sorted(s_clip2time, key=s_clip2time.get):
        # print k, s_clip2time[k], s_clip2label[k]
        if s_clip2label[k] == 1:
            preictals.append(k)
        else:
            interictals.append(k)

    preictal_groups = []
    n_preictal_groups = len(preictals) / 6.
    g = []
    for c in preictals:
        # print c
        if not g:
            g.append(c)
        elif len(g) < 6 and s_clip2time[c] - s_clip2time[g[-1]] < 1:
            g.append(c)
        else:
            # print g
            preictal_groups.append(g)
            g = [c]
    if g:
        preictal_groups.append(g)

    s = 0
    for i, g in enumerate(preictal_groups):
        # print i, g
        s += len(g)
    assert s == len(preictals)

    interictal_groups = []
    n_interictal_groups = len(interictals) / 6.
    # print n_interictal_groups
    g = []
    for c in interictals:
        if not g:
            g.append(c)
        elif len(g) < 6 and (s_clip2time[c] - s_clip2time[g[-1]]) < 1:
            g.append(c)
        else:
            interictal_groups.append(g)
            g = [c]
    if g:
        interictal_groups.append(g)

    s = 0
    for i, g in enumerate(interictal_groups):
        # print i, g
        s += len(g)
    assert s == len(interictals)

    return preictal_groups, interictal_groups


def load_grouped_train_data(data_path, subject, full_hour=False):
    grouped_filenames = utils.load_pkl('filenames.pkl')
    groups_x, groups_y, groups_t = [], [], []
    preictal_fnames = grouped_filenames[subject]['preictal']
    print 'preictal groups', len(preictal_fnames)
    interictal_fnames = grouped_filenames[subject]['interictal']
    print 'n interictal groups', len(interictal_fnames)
    groups_filenames = interictal_fnames + preictal_fnames

    d = loadmat(pathfinder.LABELS_PATH + '/' + subject + '_answer_key.mat')
    pre_times_mat = list(d['answer_key']['preictal_uutc'][0][0][0] * 2.77778e-10)
    int_times_mat = list(d['answer_key']['interictal_uutc'][0][0][0] * 2.77778e-10)
    pre_filenames = [utils.create_filename(subject, s, 'preictal') for s in range(1, len(pre_times_mat) + 1)]
    int_filenames = [utils.create_filename(subject, s, 'interictal') for s in range(1, len(int_times_mat) + 1)]
    times = pre_times_mat + int_times_mat
    clips = pre_filenames + int_filenames
    clip2time = {}
    for f, t in zip(clips, times):
        clip2time[f] = t

    for g in interictal_fnames:
        gx, gy, gt = [], [], []
        if not full_hour or (full_hour and len(g) == 6):
            for f in g:
                datum = loadmat(data_path + '/' + f)['data']
                gx.append(datum)
                gy.append(0)
                gt.append(clip2time[f.split('/')[-1]])
            groups_x.append(gx)
            groups_y.append(gy)
            groups_t.append(gt)

    for g in preictal_fnames:
        gx, gy, gt = [], [], []
        if not full_hour or (full_hour and len(g) == 6):
            for f in g:
                datum = loadmat(data_path + '/' + f)['data']
                gx.append(datum)
                gy.append(1)
                gt.append(clip2time[f.split('/')[-1]])
            groups_x.append(gx)
            groups_y.append(gy)
            groups_t.append(gt)

    return groups_x, groups_y, groups_filenames, groups_t


def load_grouped_test_data(data_path, grouped_preictal_fnames, grouped_interictal_fnames, subject, full_hour=False):
    groups_x, groups_y = [], []
    groups_filenames = grouped_preictal_fnames + grouped_interictal_fnames
    for g in grouped_preictal_fnames:
        gx, gy = [], []
        if not full_hour or (full_hour and len(g) == 6):
            for f in g:
                f = 'holdout' + f.replace(subject, '') if 'holdout' in f else f
                datum = loadmat(data_path + '/' + subject + '/' + f)['data']
                gx.append(datum)
                gy.append(1)
            groups_x.append(gx)
            groups_y.append(gy)

    for g in grouped_interictal_fnames:
        gx, gy = [], []
        if not full_hour or (full_hour and len(g) == 6):
            for f in g:
                f = 'holdout' + f.replace(subject, '') if 'holdout' in f else f
                datum = loadmat(data_path + '/' + subject + '/' + f)['data']
                gx.append(datum)
                gy.append(0)
            groups_x.append(gx)
            groups_y.append(gy)
    return groups_x, groups_y, groups_filenames


def load_holdout_data(data_path, subject):
    real_subject = subject.replace('_holdout', '')
    read_dir = data_path + '/' + subject
    all_filenames = sorted(os.listdir(read_dir))
    test_filenames = [f for f in all_filenames if 'holdout' in f]
    x = []
    filename2idx, idx2filename = {}, {}
    for i, filename in enumerate(test_filenames):
        x.append(loadmat(read_dir + '/' + filename)['data'])
        filename2idx[real_subject + '_' + filename] = i
        idx2filename[i] = real_subject + '_' + filename
    x = np.stack(x)
    return x, filename2idx, idx2filename


def load_train_data(data_path, subject, dataset='train', proportion_valid=0):
    read_dir = data_path + '/' + subject
    all_filenames = sorted(os.listdir(read_dir))
    filenames = [f for f in all_filenames if 'test' not in f]
    n_valid = int(len(filenames) * proportion_valid)
    rng = np.random.RandomState(42)
    valid_idxs = rng.choice(range(len(filenames)), n_valid, False)

    x_train, y_train = [], []
    x_valid, y_valid = [], []
    filename2idx_train, idx2filename_train = {}, {}
    filename2idx_valid, idx2filename_valid = {}, {}
    valid_idx, train_idx = 0, 0
    for i, filename in enumerate(filenames):
        if i in valid_idxs:
            x_valid.append(loadmat(read_dir + '/' + filename)['data'])
            y_valid.append(1 if 'preictal' in filename else 0)
            filename2idx_valid[filename] = valid_idx
            idx2filename_valid[valid_idx] = filename
            valid_idx += 1
        else:
            x_train.append(loadmat(read_dir + '/' + filename)['data'])
            y_train.append(1 if 'preictal' in filename else 0)
            filename2idx_train[filename] = train_idx
            idx2filename_train[train_idx] = filename
            train_idx += 1

    if dataset == 'valid':
        x_valid, y_valid = np.stack(x_valid), np.stack(y_valid)
        return x_valid, y_valid, filename2idx_valid, idx2filename_valid

    x_train, y_train = np.stack(x_train), np.stack(y_train)
    return x_train, y_train, filename2idx_train, idx2filename_train


def read_file(path, expand_dims=True):
    x = loadmat(path)['data']
    if expand_dims:
        x = np.expand_dims(x, 1)
    y = 1 if 'preictal' in path else 0
    filename = path.split('/')[-2:]
    return x, y, filename


def load_test_data(data_path, subject):
    read_dir = data_path + '/' + subject
    all_filenames = sorted(os.listdir(read_dir))
    test_filenames = [f for f in all_filenames if 'test' in f]
    x = []
    filename2idx, idx2filename = {}, {}
    for i, filename in enumerate(test_filenames):
        x.append(loadmat(read_dir + '/' + filename, squeeze_me=True)['data'])
        filename2idx[filename] = i
        idx2filename[i] = filename
    x = np.stack(x)
    return x, filename2idx, idx2filename


def scale_across_time(x, scalers=None):
    n_examples = x.shape[0]
    n_channels = x.shape[1]
    n_fbins = x.shape[2]
    n_timesteps = x.shape[3]
    if scalers is None:
        scalers = [None] * n_channels

    for i in range(n_channels):
        xi = np.transpose(x[:, i, :, :], axes=(0, 2, 1))
        xi = xi.reshape((n_examples * n_timesteps, n_fbins))

        if scalers[i] is None:
            scalers[i] = StandardScaler()
            scalers[i].fit(xi)

        xi = scalers[i].transform(xi)
        xi = xi.reshape((n_examples, n_timesteps, n_fbins))
        xi = np.transpose(xi, axes=(0, 2, 1))
        x[:, i, :, :] = xi
    return x, scalers
