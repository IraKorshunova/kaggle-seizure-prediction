import matplotlib
import sklearn

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np
import loader
import pathfinder
import configuration
import utils
import sklearn.metrics
import os


def train_predictions_plot(predictions_exp_dir, subjects):
    train_clip2label = loader.load_train_labels(pathfinder.LABELS_PATH)

    x, y, labels = [], [], []
    for i, subject in enumerate(subjects):
        id2pred = utils.load_pkl(predictions_exp_dir + '/%s-train.pkl' % subject)
        subject_prob, subject_labels = [], []
        pp, ip = [], []
        for id, pred in id2pred.iteritems():
            subject_prob.append(pred)
            id = os.path.basename(id)
            subject_labels.append(train_clip2label[id])
            if train_clip2label[id] == 0:
                ip.append(pred)
            else:
                pp.append(pred)

        print subject
        print '\n Train loss:', utils.cross_entropy_loss(subject_labels, subject_prob)

        y.extend(subject_prob)
        labels.extend(subject_labels)
        x.extend([i] * len(subject_prob))

    x = np.array(x, dtype='float32')
    y = np.array(y)
    # add jitter
    rng = np.random.RandomState(42)
    x += rng.normal(0.0, 0.08, size=len(x))

    color = ['b' if l == 0 else 'r' for l in labels]
    markers = ['.' if l == 0 else '^' for l in labels]

    zip_all = sorted(zip(x, y, color, markers), key=lambda tup: tup[2])

    plt.figure()
    for a_, b_, c_, d_ in zip_all:
        plt.scatter(a_, b_, c=c_, s=60, marker=d_)

    plt.ylabel('Preictal probability', fontsize=20)
    plt.xticks(range(0, 7), subjects)
    plt.subplots_adjust(bottom=0.15)
    plt.legend()
    ax = plt.gca()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)
    plt.xlim([-0.5, 6.5])
    plt.ylim([-0.1, 1.1])
    ax = plt.subplot(111)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=2, fancybox=True, shadow=True)

    plt.savefig(img_path + '/' + config_name + '-predictions-train.png')


def test_predictions_plot(predictions_exp_dir, subjects):
    test_clip2label, test_clip2time, test_clip2usage = loader.load_test_labels(pathfinder.LABELS_PATH)

    x, y, labels = [], [], []
    for i, subject in enumerate(subjects):
        id2pred = utils.load_pkl(predictions_exp_dir + '/%s-test.pkl' % subject)
        subject_prob, subject_labels = [], []
        pp, ip = [], []
        for id, pred in id2pred.iteritems():
            subject_prob.append(pred)
            subject_labels.append(test_clip2label[id])
            if test_clip2label[id] == 0:
                ip.append(pred)
            else:
                pp.append(pred)
        print subject
        fpr, tpr, threshold = sklearn.metrics.roc_curve(subject_labels, subject_prob)
        c = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
        opt_threshold = threshold[np.where(c == np.min(c))]
        print 'threshold', opt_threshold

        y.extend(subject_prob)
        labels.extend(subject_labels)
        x.extend([i] * len(subject_prob))

    x = np.array(x, dtype='float32')
    y = np.array(y)
    # add jitter
    rng = np.random.RandomState(42)
    x += rng.normal(0.0, 0.08, size=len(x))

    color = ['b' if l == 0 else 'r' for l in labels]
    markers = ['.' if l == 0 else '^' for l in labels]

    zip_all = sorted(zip(x, y, color, markers), key=lambda tup: tup[2])
    plt.figure()
    for a_, b_, c_, d_ in zip_all:
        plt.scatter(a_, b_, c=c_, s=60, marker=d_)

    plt.scatter(zip_all[0][0], zip_all[0][1], c=zip_all[0][2], s=60, marker=zip_all[0][3], label='interictal')
    plt.scatter(zip_all[-1][0], zip_all[-1][1], c=zip_all[-1][2], s=60, marker=zip_all[-1][3], label='preictal')

    plt.ylabel('Preictal probability', fontsize=20)
    plt.xticks(range(0, 7), subjects)
    plt.subplots_adjust(bottom=0.15)
    plt.legend()
    ax = plt.gca()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)
    plt.xlim([-0.5, 6.5])
    plt.ylim([-0.1, 1.1])
    ax = plt.subplot(111)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=2, fancybox=True, shadow=True)

    plt.savefig(img_path + '/' + config_name + '-predictions-test.png')
    plt.show()



def holdout_predictions_plot(predictions_exp_dir, subjects):
    train_clip2label, _ = loader.load_holdout_labels(pathfinder.LABELS_PATH)

    x, y, labels = [], [], []
    for i, subject in enumerate(subjects):
        id2pred = utils.load_pkl(predictions_exp_dir + '/%s.pkl' % subject)
        subject_prob, subject_labels = [], []
        pp, ip = [], []
        for id, pred in id2pred.iteritems():
            subject_prob.append(pred)
            id = os.path.basename(id)
            subject_labels.append(train_clip2label[id])
            if train_clip2label[id] == 0:
                ip.append(pred)
            else:
                pp.append(pred)

        print subject
        fpr, tpr, threshold = sklearn.metrics.roc_curve(subject_labels, subject_prob)
        c = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
        opt_threshold = threshold[np.where(c == np.min(c))]
        print 'threshold', opt_threshold
        print '\n Holdout loss:', utils.cross_entropy_loss(subject_labels, subject_prob)

        y.extend(subject_prob)
        labels.extend(subject_labels)
        x.extend([i] * len(subject_prob))

    x = np.array(x, dtype='float32')
    y = np.array(y)
    # add jitter
    rng = np.random.RandomState(42)
    x += rng.normal(0.0, 0.08, size=len(x))

    color = ['b' if l == 0 else 'r' for l in labels]
    markers = ['.' if l == 0 else '^' for l in labels]

    zip_all = sorted(zip(x, y, color, markers), key=lambda tup: tup[2])
    plt.figure()
    for a_, b_, c_, d_ in zip_all:
        plt.scatter(a_, b_, c=c_, s=60, marker=d_)

    plt.ylabel('Preictal probability', fontsize=20)
    plt.xticks(range(0, 7), subjects)
    plt.subplots_adjust(bottom=0.15)
    plt.legend()
    ax = plt.gca()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)
    plt.xlim([-0.5, 6.5])
    plt.ylim([-0.1, 1.1])
    ax = plt.subplot(111)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=2, fancybox=True, shadow=True)

    # plt.savefig(img_path + '/' + config_name + '-predictions-holdout.png')
    # plt.savefig(img_path + '/' + config_name + '-predictions-holdout.png')
    plt.show()

if __name__ == '__main__':

    config_name = 'n8b1m_relu_sm'
    # config_name = 'lda_8b1m'
    # config_name = 'svm_rbf_c10'
    # config_name = 'svm_hrbf_c10'


    configuration.set_configuration(config_name)

    img_path = pathfinder.IMG_PATH
    prediction_dir = utils.get_dir_path('predictions', pathfinder.METADATA_PATH)
    predictions_exp_dir = utils.find_model_metadata(prediction_dir, config_name)
    subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    # subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5']
    test_predictions_plot(predictions_exp_dir, subjects)
    # train_predictions_plot(predictions_exp_dir, subjects)
    # subjects = ['Dog_1_holdout', 'Dog_2_holdout', 'Dog_3_holdout', 'Dog_4_holdout']
    # holdout_predictions_plot(predictions_exp_dir, subjects)
