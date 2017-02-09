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

    ppv_test_cnn = d_cnn['ppv_test'][subject]
    ppv_test_lda = d_lda['ppv_test'][subject]
    ppv_test_svm = d_svm['ppv_test'][subject]

    sp_test_cnn = d_cnn['sp_test'][subject]
    sp_test_lda = d_lda['sp_test'][subject]
    sp_test_svm = d_svm['sp_test'][subject]

    sn_hd_cnn = d_cnn['sn_holdout'][subject]
    sn_hd_lda = d_lda['sn_holdout'][subject]
    sn_hd_svm = d_svm['sn_holdout'][subject]

    ppv_hd_cnn = d_cnn['ppv_holdout'][subject]
    ppv_hd_lda = d_lda['ppv_holdout'][subject]
    ppv_hd_svm = d_svm['ppv_holdout'][subject]

    sp_hd_cnn = d_cnn['sp_holdout'][subject]
    sp_hd_lda = d_lda['sp_holdout'][subject]
    sp_hd_svm = d_svm['sp_holdout'][subject]

    t_lda_cv = d_lda['t_cv_opt'][subject]
    t_svm_cv = d_svm['t_cv_opt'][subject]

    thresholds = d_cnn['thresholds']
    ss = 120
    lw = 2.0
    ymin = -7
    ymax = 107
    xmin = -0.01
    xmax = 0.8

    fig = plt.figure(figsize=(10, 10))

    # PPV
    plt.subplot(311)
    plt.plot(thresholds, ppv_test_cnn, label='set 1', linewidth=3.5, c=c_cnn, zorder=4, alpha=0.6)
    plt.plot(thresholds, ppv_test_svm, label='set 1', linewidth=3.5, c=c_svm, alpha=1.)
    plt.plot(thresholds, ppv_test_lda, label='set 1', linewidth=3.5, c=c_lda, alpha=1.)

    plt.plot(thresholds, ppv_hd_cnn, label='set 2 CNN', linewidth=3.5, ls='dashed', c=c_cnn, zorder=4, alpha=0.6)
    plt.plot(thresholds, ppv_hd_svm, label='set 2 SVM', linewidth=3.5, ls='dashed', c=c_svm)
    plt.plot(thresholds, ppv_hd_lda, label='set 2 LDA', linewidth=3.5, ls='dashed', c=c_lda)

    plt.vlines(x=t_lda_cv, ymin=ymin, ymax=ymax, color=c_lda, linewidth=lw)
    plt.vlines(x=t_svm_cv, ymin=ymin, ymax=ymax, color=c_svm, linewidth=lw)
    plt.vlines(x=0.5, ymin=ymin, ymax=ymax, color='black', linewidth=1., linestyles='-')

    ax = plt.subplot(311)
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.yticks([0, 25, 50, 75, 100])
    plt.ylabel([0, 25, 50, 75, 100])
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    plt.ylabel('PPV', fontsize=20)
    ax.xaxis.set_major_formatter(plt.NullFormatter())

    # SENSITIVITY
    plt.subplot(312)
    plt.plot(thresholds, sn_test_cnn, label='set 1', linewidth=3.5, c=c_cnn, zorder=4, alpha=0.6)
    plt.plot(thresholds, sn_test_svm, label='set 1', linewidth=3.5, c=c_svm, alpha=1.)
    plt.plot(thresholds, sn_test_lda, label='set 1', linewidth=3.5, c=c_lda, alpha=1.)

    plt.plot(thresholds, sn_hd_cnn, label='set 2 CNN', linewidth=3.5, ls='dashed', c=c_cnn, zorder=4, alpha=0.6)
    plt.plot(thresholds, sn_hd_svm, label='set 2 SVM', linewidth=3.5, ls='dashed', c=c_svm)
    plt.plot(thresholds, sn_hd_lda, label='set 2 LDA', linewidth=3.5, ls='dashed', c=c_lda)

    plt.vlines(x=t_lda_cv, ymin=ymin, ymax=ymax, color=c_lda, linewidth=lw)
    plt.vlines(x=t_svm_cv, ymin=ymin, ymax=ymax, color=c_svm, linewidth=lw)
    plt.vlines(x=0.5, ymin=ymin, ymax=ymax, color='black', linewidth=1., linestyles='-')
    # plt.vlines(x=t_lda_holdout, ymin=ymin, ymax=ymax, color=c_lda, linewidth=lw, linestyles='dashed')
    # plt.vlines(x=t_svm_holdout, ymin=ymin, ymax=ymax, color=c_svm, linewidth=lw, linestyles='dashed')

    ax = plt.subplot(312)
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.yticks([0, 25, 50, 75, 100])
    plt.ylabel([0, 25, 50, 75, 100])
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    plt.ylabel('TPR', fontsize=20)
    ax.xaxis.set_major_formatter(plt.NullFormatter())

    # SPECIFICITY
    plt.subplot(313)
    plt.plot(thresholds, sp_test_cnn, label='test', linewidth=3.5, c=c_cnn, zorder=3, alpha=0.6)
    plt.plot(thresholds, sp_test_svm, label='test', linewidth=3.5, c=c_svm)
    plt.plot(thresholds, sp_test_lda, label='test', linewidth=3.5, c=c_lda)
    plt.plot(thresholds, sp_hd_cnn, label='hold-out CNN', linewidth=3.5, ls='dashed', c=c_cnn, zorder=3, alpha=0.6)
    plt.plot(thresholds, sp_hd_svm, label='hold-out SVM', linewidth=3.5, ls='dashed', c=c_svm)
    plt.plot(thresholds, sp_hd_lda, label='hold-out LDA', linewidth=3.5, ls='dashed', c=c_lda)

    y_pos_lda = 2. / 3 * ymin
    y_pos_svm = y_pos_lda
    if t_lda_cv - 0.01 < t_svm_cv < t_lda_cv + 0.01:
        y_pos_svm = 1. / 3 * ymin

    s1 = plt.scatter(t_svm_cv, y_pos_svm, c=c_svm, zorder=11, s=ss)
    s2 = plt.scatter(t_lda_cv, y_pos_lda, c=c_lda, zorder=10, s=ss)

    plt.vlines(x=t_lda_cv, ymin=0, ymax=ymax, color=c_lda, linewidth=lw)
    plt.vlines(x=t_svm_cv, ymin=0, ymax=ymax, color=c_svm, linewidth=lw)

    # plt.hlines(xmin=-0.01, xmax=1.0, y=75, color='black', linewidth=1., zorder=20, linestyles='-')
    plt.vlines(x=0.5, ymin=0, ymax=ymax, color='black', linewidth=1., linestyles='-')

    ax = plt.subplot(313)
    ax.fill_between(np.arange(xmin, xmax, 0.001), ymin - 1.5, 0, facecolor='gray', alpha=0.3)

    l_pos = [2.4, 1.5, 2.55, 2.55]

    l1 = ax.legend(loc='lower right', fancybox=True, fontsize=16, ncol=2, columnspacing=1,
                   handletextpad=0.15, numpoints=1, bbox_to_anchor=(1.01, 2.58))

    ax.legend([s1, s2], ['threshold SVM', 'threshold LDA'], loc='lower right',
              fancybox=True, fontsize=16, ncol=1, columnspacing=1,
              handletextpad=0.15, scatterpoints=1, bbox_to_anchor=(1.01, 0.06))
    plt.gca().add_artist(l1)

    plt.xlim([xmin, xmax])
    plt.ylim([ymin - 1.5, ymax])
    plt.yticks([0, 25, 50, 75, 100])
    plt.ylabel([0, 25, 50, 75, 100])
    plt.xticks(np.arange(0, xmax + 0.1, 0.1))
    plt.xlabel(np.arange(0, xmax + 0.1, 0.1))

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    plt.ylabel('TNR', fontsize=20)
    plt.xlabel('Threshold', fontsize=20)

    fig.subplots_adjust(wspace=0, hspace=0.05)
    plt.savefig('../metadata/images/%s-spsn.eps' % subject, format='eps')
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
    # subjects = ['Dog_1_holdout', 'Dog_4_holdout']
    # subjects = ['Dog_4_holdout']
    for s in subjects:
        plot_subject(s, d_cnn, d_lda, d_svm, False)
