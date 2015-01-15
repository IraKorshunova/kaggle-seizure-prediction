import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from test_labels_loader import load_test_labels


def minmax_rescale(probability):
    scaler = MinMaxScaler(feature_range=(0.000000001, 0.999999999))
    return scaler.fit_transform(probability)


def softmax_rescale(probability):
    norm_x = StandardScaler().fit_transform(probability)
    return 1.0 / (1.0 + np.exp(-norm_x))


def plot(clips, probs, labels, scale=None):
    x, y = [], []
    for i, subject in enumerate(subjects):
        subject_idx = []
        for j, s in enumerate(clips):
            if subject in s:
                subject_idx.append(j)
        subject_idx = np.array(subject_idx)
        subj_prob = probs[subject_idx]

        if scale == 'softmax':
            y.extend(softmax_rescale(np.expand_dims(subj_prob, axis=1)))
        elif scale == 'minmax':
            y.extend(minmax_rescale(np.expand_dims(subj_prob, axis=1)))
        else:
            y.extend(subj_prob)

        x.extend([i] * len(subject_idx))

    x = np.array(x, dtype='float32')
    y = np.array(y)
    # add jitter
    rng = np.random.RandomState(42)
    x += rng.normal(0.0, 0.08, size=len(x))

    color = ['b'] * len(y)
    markers = ['.'] * len(y)

    if labels:
        color = []
        for subject in subjects:
            subj_lables = labels[subject]['preictal']
            color.extend(map(str, subj_lables))

        color = np.array(color)
        markers = np.copy(color)
        markers[np.where(markers == '0')[0]] = '+'
        markers[np.where(markers == '1')[0]] = '^'

        color[np.where(color == '0')[0]] = 'b'
        color[np.where(color == '1')[0]] = 'r'

    zip_all = sorted(zip(x, y, color, markers), key=lambda tup: tup[2])

    for a_, b_, c_, d_ in zip_all:
        plt.scatter(a_, b_, c=c_, s=60, marker=d_)

    if labels:
        x1 = zip_all[0]
        plt.scatter(x1[0], x1[1], c=x1[2], s=60, marker=x1[3], label='interictal')
        x1 = zip_all[-1]
        plt.scatter(x1[0], x1[1], c=x1[2], s=60, marker='^', label='preictal')

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

    plt.show()


if __name__ == '__main__':
    test_labels_path = '/mnt/sda4/CODING/python/kaggle_data/test_labels.csv'
    s1 = '/mnt/sda4/CODING/python/kaggle_data/submission_0.78612.csv'
    s2 = '/mnt/sda4/CODING/python/kaggle_data/submission_lda8.csv'
    subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']

    submission_df = read_csv(s1)
    probs = submission_df['preictal']
    clips = submission_df['clip']
    labels = load_test_labels(test_labels_path)

    plot(clips, probs, labels)


