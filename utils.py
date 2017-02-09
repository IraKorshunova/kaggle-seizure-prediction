import cPickle
import os
import platform
import subprocess
import time
import glob
import sklearn.metrics
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
import theano.tensor as T


def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def auc_sklearn(targets, predictions):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(targets, predictions, pos_label=1, drop_intermediate=False)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    return roc_auc


def get_tpr(targets, predictions, tnr=0.75):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(targets, predictions, pos_label=1, drop_intermediate=False)
    spc = np.around(1. - fpr, 2)
    closest = np.argwhere(spc == tnr)
    return np.mean(tpr[closest]), np.mean(thresholds[closest])


def cross_entropy_loss(targets, predictions):
    targets = np.array(targets)
    predictions = np.array(predictions)
    predictions = predictions.clip(1e-6, 0.999999)
    ce = -targets * np.log(predictions) - (1. - targets) * np.log(1 - predictions)
    return np.mean(ce)


def get_filtered_data_name(transform_params):
    highcut = transform_params['highcut']
    lowcut = transform_params['lowcut']
    config_name = 'lowcut' + str(lowcut) + 'highcut' + str(highcut)
    return config_name


def get_fft_data_name(transform_params):
    return 'fft_' + transform_params['features'] + '_' + get_filtered_data_name(transform_params) \
           + 'nfreq_bands' + str(transform_params['nfreq_bands']) \
           + 'win_length_sec' + str(transform_params['win_length_sec']) \
           + 'stride_sec' + str(transform_params['stride_sec'])


def find_model_metadata(metadata_dir, config_name):
    metadata_paths = glob.glob(metadata_dir + '/%s-*' % config_name)
    if not metadata_paths:
        raise ValueError('No metadata files for config %s' % config_name)
    elif len(metadata_paths) > 1:
        print metadata_paths
        raise ValueError('Multiple metadata files for config %s' % config_name)
    return metadata_paths[0]


def get_dir_path(dir_name, root_dir):
    dir_path = root_dir + '/' + dir_name
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return dir_path


def get_subject_records(all_subjects_dict, subject_id):
    new_dict = {}
    for k in all_subjects_dict.iterkeys():
        if subject_id in k:
            new_dict[k] = all_subjects_dict[k]
    return new_dict


def get_subject_id(clip_id):
    return '_'.join(clip_id.split('_', 2)[:2])


def hms(seconds):
    seconds = np.floor(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    return "%02d:%02d:%02d" % (hours, minutes, seconds)


def timestamp():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def hostname():
    return platform.node()


def generate_expid(arch_name):
    return "%s-%s-%s" % (arch_name, hostname(), timestamp())


def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
    except:
        return 0


def save_pkl(obj, path, protocol=cPickle.HIGHEST_PROTOCOL):
    with open(path, 'w') as f:
        cPickle.dump(obj, f, protocol=protocol)


def load_pkl(path):
    with open(path) as f:
        obj = cPickle.load(f)
    return obj


def copy(from_folder, to_folder):
    command = "cp -r %s %s/." % (from_folder, to_folder)
    print command
    os.system(command)


def current_learning_rate(schedule, idx):
    s = schedule.keys()
    s.sort()
    current_lr = schedule[0]
    for i in s:
        if idx >= i:
            current_lr = schedule[i]

    return current_lr


def reshape_data_for_lda(x, y=None):
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


def make_submission_file(predictions_exp_dir, submission_path, subjects):
    prediction_files = [predictions_exp_dir + '/%s-test.pkl' % s for s in subjects]
    with open(submission_path, 'wb') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['clip', 'preictal'])
        for p in prediction_files:
            id2prediction = load_pkl(p)
            for k in sorted(id2prediction):
                csv_writer.writerow([k, str(id2prediction[k])])

    print ' submission saved to %s' % submission_path


def create_filename(subject, n, label):
    return subject + '_' + label + '_segment' + '_' + str(n).zfill(4) + '.mat'


def softmax_scaler(x, scaler=None):
    if not scaler:
        norm_x = StandardScaler().fit_transform(x)
    else:
        norm_x = scaler.transform(x)
    return 1.0 / (1.0 + np.exp(-norm_x))


def minmax_scaler(x):
    return (x - min(x)) / (max(x) - min(x))


def norm(x, L=2, axis=None, keepdims=False):
    # optimizations will/should catch cases like L=1, L=2
    y = T.basic.pow(
        T.basic.pow(
            T.basic.abs_(x), L).sum(axis=axis), 1.0 / L)
    if keepdims:
        return T.basic.makeKeepDims(x, y, axis)
    else:
        return y
