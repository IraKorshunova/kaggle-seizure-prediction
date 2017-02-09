import matplotlib

matplotlib.use('Agg')
import sys
import numpy as np
import utils
from configuration import config, set_configuration
import pathfinder
import preprocess
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import loader
import matplotlib.pyplot as plt


def predict_test(subject, model, data_scaler, predictions_scaler=None):
    predictions_path_test = predictions_exp_dir + '/' + subject + '-test.pkl'
    data_path = preprocess.preprocess_data(pathfinder.RAW_DATA_PATH, subject, config().transformation_params)
    x_test, _, idx2filename = loader.load_test_data(data_path, subject)
    n_test_examples = x_test.shape[0]
    n_timesteps = x_test.shape[3]

    x_test = utils.reshape_data_for_lda(x_test)
    x_test = data_scaler.transform(x_test)

    pred_1m = model.predict_proba(x_test)[:, 1]

    pred_10m = np.reshape(pred_1m, (n_test_examples, n_timesteps))
    pred_10m = np.mean(pred_10m, axis=1)
    if config().transformation_params.get('calibr'):
        pred_10m = utils.softmax_scaler(pred_10m)
        print 'softmax'
    if config().transformation_params.get('calibr_train'):
        pred_10m = utils.softmax_scaler(pred_10m, predictions_scaler)
        print 'softmax train'

    id2prediction = {}
    for i in xrange(len(pred_10m)):
        id2prediction[idx2filename[i]] = pred_10m[i]
    utils.save_pkl(id2prediction, predictions_path_test)

    if config().transformation_params.get('hours'):
        clip2label, clip2time, clip2usage = loader.load_test_labels(pathfinder.LABELS_PATH)
        test_preictal_groups, test_interictal_groups = loader.group_labels_by_hour(clip2label, clip2time, subject)
        test_groups = test_preictal_groups + test_interictal_groups

        groupid2prediction = {}
        for i, group in enumerate(test_groups):
            if len(group) != 6:
                print 'skipping group with length', len(group)
            elif len(group) == 6:
                group_prediction = np.mean([id2prediction[clip] for clip in group])
                group_id = group[0]
                groupid2prediction[group_id] = group_prediction

        utils.save_pkl(groupid2prediction, predictions_path_test)


def predict_train(subject, model, data_scaler):
    predictions_path_test = predictions_exp_dir + '/' + subject + '-train.pkl'
    data_path = preprocess.preprocess_data(pathfinder.RAW_DATA_PATH, subject, config().transformation_params)
    x_train, _, filename2idx, idx2filename = loader.load_train_data(data_path, subject)
    n_test_examples = x_train.shape[0]
    n_timesteps = x_train.shape[3]

    x_train = utils.reshape_data_for_lda(x_train)
    x_train = data_scaler.transform(x_train)

    pred_1m = model.predict_proba(x_train)[:, 1]

    pred_10m = np.reshape(pred_1m, (n_test_examples, n_timesteps))
    pred_10m = np.mean(pred_10m, axis=1)
    id2prediction = {}
    for i in xrange(len(pred_10m)):
        id2prediction[idx2filename[i]] = pred_10m[i]
    utils.save_pkl(id2prediction, predictions_path_test)
    return StandardScaler().fit(pred_10m)


def predict_holdout(subject, model, data_scaler, predictions_scaler):
    predictions_path_test = predictions_exp_dir + '/' + subject + '.pkl'
    data_path = preprocess.preprocess_data(pathfinder.RAW_DATA_PATH, subject, config().transformation_params)
    x_test, _, idx2filename = loader.load_holdout_data(data_path, subject)
    n_test_examples = x_test.shape[0]
    n_timesteps = x_test.shape[3]

    x_test = utils.reshape_data_for_lda(x_test)
    x_test = data_scaler.transform(x_test)

    pred_1m = model.predict_proba(x_test)[:, 1]

    pred_10m = np.reshape(pred_1m, (n_test_examples, n_timesteps))
    pred_10m = np.mean(pred_10m, axis=1)

    if config().transformation_params.get('calibr'):
        pred_10m = utils.softmax_scaler(pred_10m)
        print 'softmax'
    if config().transformation_params.get('calibr_train'):
        pred_10m = utils.softmax_scaler(pred_10m, predictions_scaler)
        print 'softmax train'

    id2prediction = {}
    for i in xrange(len(pred_10m)):
        id2prediction[idx2filename[i]] = pred_10m[i]
    utils.save_pkl(id2prediction, predictions_path_test)

    if config().transformation_params.get('hours'):
        clip2label, clip2time = loader.load_holdout_labels(pathfinder.LABELS_PATH)
        holdout_preictal_groups, holdout_interictal_groups = loader.group_labels_by_hour(
            clip2label, clip2time, subject)

        test_groups = holdout_preictal_groups + holdout_interictal_groups

        groupid2prediction = {}
        for i, group in enumerate(test_groups):
            if len(group) != 6:
                print 'skipping group with length', len(group)
            elif len(group) == 6:
                group_prediction = np.mean([id2prediction[clip] for clip in group])
                group_id = group[0]
                groupid2prediction[group_id] = group_prediction

        utils.save_pkl(groupid2prediction, predictions_path_test)


def train(subject, plot=False):
    print 'Load data'
    data_path = preprocess.preprocess_data(pathfinder.RAW_DATA_PATH, subject, config().transformation_params)
    x, y, _, _ = loader.load_train_data(data_path, subject)
    print
    print 'Data'
    print 'n_clips', len(y)
    print '% interictal', -1. * np.sum(y - 1) / len(y)
    print 'n_channels', x.shape[1]
    n_channels = x.shape[1]
    n_fbins = x.shape[2]

    x, y = utils.reshape_data_for_lda(x, y)
    data_scaler = StandardScaler()
    x = data_scaler.fit_transform(x)

    model = LinearDiscriminantAnalysis()
    model.fit(x, y)
    coef = model.coef_[:1].T
    channels = []
    fbins = []
    for c in range(n_channels):
        fbins.extend(range(n_fbins))  # 0- delta, 1- theta ...
        channels.extend([c] * n_fbins)

    sigma_x = np.cov(x, rowvar=0)
    y = np.expand_dims(y, axis=1)
    sigma_y_inv = 1. / np.cov(y, rowvar=0)
    a = np.dot(sigma_x, coef) * sigma_y_inv

    if plot:
        fig = plt.figure()
        for i in range(n_channels):
            if n_channels == 24:
                fig.add_subplot(4, 6, i + 1)
            else:
                fig.add_subplot(4, 4, i + 1)
            ax = plt.gca()
            ax.set_xlim([0, n_fbins])
            ax.set_xticks(np.arange(0.5, n_fbins + 0.5, 1))
            ax.set_xticklabels(np.arange(0, n_fbins))
            max_y = max(a) + 0.01
            min_y = min(a) - 0.01
            ax.set_ylim([0, max_y])
            ax.set_yticks(np.around(np.arange(min_y, max_y, (max_y - min_y) / 4.0), decimals=1))
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(15)
            plt.bar(range(0, n_fbins), a[i * n_fbins:i * n_fbins + n_fbins])
        fig.suptitle(subject, fontsize=20)
        plt.savefig(subject + '.png')
    return model, data_scaler


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit("Usage: train.py <configuration_name>")

    config_name = sys.argv[1]
    set_configuration(config_name)
    expid = utils.generate_expid(config_name)
    print
    print "Experiment ID: %s" % expid
    print

    # predictions paths
    prediction_dir = utils.get_dir_path('predictions', pathfinder.METADATA_PATH)
    predictions_exp_dir = prediction_dir + "/%s" % expid
    utils.make_dir(predictions_exp_dir)

    # submissions paths
    submission_dir = utils.get_dir_path('submissions', pathfinder.METADATA_PATH)
    submission_path = submission_dir + "/%s.csv" % expid

    if len(sys.argv) == 3:
        subjects = [sys.argv[2]]
    else:
        subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    for subject in subjects:
        print '***********************', subject, '***************************'
        model, data_scaler = train(subject)
        predictions_scaler = predict_train(subject, model, data_scaler)
        predict_test(subject, model, data_scaler, predictions_scaler)
        if subject in ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4']:
            predict_holdout(subject + '_holdout', model, data_scaler, predictions_scaler)

    utils.make_submission_file(predictions_exp_dir, submission_path, subjects)
