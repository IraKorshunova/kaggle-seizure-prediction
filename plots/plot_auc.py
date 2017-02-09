import numpy as np
import matplotlib.pyplot as plt
import utils

c_cnn = '#ff3333'
c_lda = '#2c7fb8'
c_svm = '#7fcdbb'


def plot_subject(subject, d_cnn, d_lda, d_svm, sp_curve=True):
    sn_test_cnn = d_cnn['sn_test'][subject]
    sn_test_lda = d_lda['sn_test'][subject]
    sn_test_svm = d_svm['sn_test'][subject]

    sp_test_cnn = d_cnn['sp_test'][subject]
    sp_test_lda = d_lda['sp_test'][subject]
    sp_test_svm = d_svm['sp_test'][subject]

    sn_hd_cnn = d_cnn['sn_holdout'][subject]
    sn_hd_lda = d_lda['sn_holdout'][subject]
    sn_hd_svm = d_svm['sn_holdout'][subject]

    sp_hd_cnn = d_cnn['sp_holdout'][subject]
    sp_hd_lda = d_lda['sp_holdout'][subject]
    sp_hd_svm = d_svm['sp_holdout'][subject]

    # t_lda_train = d_lda['t_train'][subject]
    # t_lda_test = d_lda['t_test'][subject]
    # t_lda_holdout = d_lda['t_holdout'][subject]
    #
    # t_svm_train = d_svm['t_train'][subject]
    # t_svm_test = d_svm['t_test'][subject]
    # t_svm_holdout = d_svm['t_holdout'][subject]

    thresholds = d_cnn['thresholds']
    ss = 120
    lw = 2.0
    ymin = -0.01
    ymax = 110
    xmin = -0.01
    xmax = 110

    fig = plt.figure(figsize=(10, 10))
    plt.subplot(111)

    plt.plot(sp_test_cnn, sn_test_cnn, c=c_cnn)
    plt.plot(sp_hd_cnn, sn_hd_cnn, c=c_cnn, ls='--')

    plt.plot(sp_test_svm, sn_test_svm, c=c_svm)
    plt.plot(sp_hd_svm, sn_hd_svm, c=c_svm, ls='--')
    print sp_test_svm
    print sn_test_svm

    plt.plot(sp_test_lda, sn_test_lda, c=c_lda)
    plt.plot(sp_hd_lda, sn_hd_lda, c=c_lda, ls='--')

    ax = plt.subplot(111)
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.yticks([0, 25, 50, 75, 100])
    plt.ylabel([0, 25, 50, 75, 100])
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    plt.ylabel('Sensitivity', fontsize=20)
    ax.xaxis.set_major_formatter(plt.NullFormatter())

    ax = plt.subplot(111)
    ax.fill_between(np.arange(xmin, xmax, 0.001), ymin - 1.5, 0, facecolor='gray', alpha=0.3)

    # l1 = ax.legend(loc='lower right', fancybox=True, fontsize=16, ncol=2, columnspacing=1,
    #           handletextpad=0.15, numpoints=1, bbox_to_anchor=(1.01, 1.65))
    # ax.legend([s2_svm, s2, s1_svm, s1], ['set 1' , 'set 1', 'set 2 SVM' , 'set 2 LDA'], loc='lower right',
    #           fancybox=True, fontsize=16, ncol=2, columnspacing=1,
    #           handletextpad=0.15, scatterpoints=1, bbox_to_anchor=(1.01, 0.06))
    # plt.gca().add_artist(l1)

    plt.xlim([xmin, xmax])
    plt.ylim([ymin - 1.5, ymax])
    plt.yticks([0, 25, 50, 75, 100])
    plt.ylabel([0, 25, 50, 75, 100])
    plt.xticks(np.arange(0, xmax + 0.1, 0.1))
    plt.xlabel(np.arange(0, xmax + 0.1, 0.1))

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    plt.ylabel('Specificity', fontsize=20)
    plt.xlabel('Threshold', fontsize=20)

    fig.subplots_adjust(wspace=0, hspace=0.01)
    # plt.tight_layout()
    plt.savefig('../metadata/images//%s-spsn.png' % subject)
    plt.show()
    return


if __name__ == '__main__':
    cnn_path = '/mnt/sda3/CODING/python/kaggle-seizure-predict/metadata/images/h8b1m-sp_sn.pkl'
    lda_path = '/mnt/sda3/CODING/python/kaggle-seizure-predict/metadata/images/lda_h8b1m-sp_sn.pkl'
    svm_path = '/mnt/sda3/CODING/python/kaggle-seizure-predict/metadata/images/svm_hrbf_c10-sp_sn.pkl'
    d_cnn = utils.load_pkl(cnn_path)
    d_lda = utils.load_pkl(lda_path)
    d_svm = utils.load_pkl(svm_path)

    subjects = ['Dog_1_holdout', 'Dog_2_holdout', 'Dog_3_holdout', 'Dog_4_holdout']

    for s in subjects:
        plot_subject(s, d_cnn, d_lda, d_svm, False)
