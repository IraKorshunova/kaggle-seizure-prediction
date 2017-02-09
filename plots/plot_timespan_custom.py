from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from loader import load_holdout_labels, load_test_labels
import utils

# subjects = ['Dog_4_holdout', 'Dog_3_holdout', 'Dog_2_holdout', 'Dog_1_holdout']
subjects = ['Dog_4_holdout', 'Dog_1_holdout']
fig = plt.figure(figsize=(10, 3))
ax = fig.add_subplot(111)
c_test = '#7fcdbb'
c_holdout = '#2c7fb8'
predictions_exp_dir = '/mnt/sda3/CODING/python/kaggle-seizure-predict/metadata/predictions/lda_h8b1m-schaap-20160608-180225'
predictions_exp_dir = '/mnt/sda3/CODING/python/kaggle-seizure-predict/metadata/predictions/h8b1m-schaap-20160525-114929'


def timesplit_test_holdout_predictions(predictions_exp_dir, holdout_subject):
    print holdout_subject
    clip2label, clip2time = load_holdout_labels('/mnt/sda3/data/kaggle-seizure-prediction/labels')
    prediction_file = predictions_exp_dir + '/%s.pkl' % holdout_subject
    clip2prediction = utils.load_pkl(prediction_file)
    print len(clip2prediction)
    hclips, htimes, hpredictions, hlabels = [], [], [], []

    for k, v in clip2prediction.iteritems():
        hpredictions.append(v)
        hclips.append(k)
        hlabels.append(clip2label[k])
        htimes.append(clip2time[k])
    usage = ['holdout'] * len(hclips)

    clip2label, clip2time, _ = load_test_labels('/mnt/sda3/data/kaggle-seizure-prediction/labels')
    subject_real = holdout_subject.replace('_holdout', '')
    prediction_file = predictions_exp_dir + '/%s-test.pkl' % subject_real
    clip2prediction = utils.load_pkl(prediction_file)
    print len(clip2prediction)

    tclips, ttimes, tpredictions, tlabels = [], [], [], []

    for k, v in clip2prediction.iteritems():
        tpredictions.append(v)
        tclips.append(k)
        tlabels.append(clip2label[k])
        ttimes.append(clip2time[k])
        usage.append('test')

    clips = hclips + tclips
    times = htimes + ttimes
    predictions = hpredictions + tpredictions
    labels = hlabels + tlabels

    z = zip(clips, times, predictions, labels, usage)
    z.sort(key=lambda x: x[1])
    last_test_idx = 0
    for i, (c, t, p, l, u) in enumerate(z):
        # print c, t, p, l, u
        if u == 'test':
            last_test_idx = i

    half = last_test_idx if 'Dog_1' in holdout_subject or 'Dog_4' in holdout_subject  else len(z) / 2
    return z[:half], z[half:]


zs = zip(np.arange(0, 4.0, 0.5), subjects)
for i, s in zs:
    print s, '======================'
    z_test, z_holdout = timesplit_test_holdout_predictions(predictions_exp_dir, s)
    htimes, ttimes = [], []
    hlabels, tlabels = [], []

    for c, t, p, l, u in z_test:
        ttimes.append(t)
        tlabels.append('set 1')
    # print 'test 0', ttimes[0]
    # print 'test -1', ttimes[-1]
    # print (ttimes[-1] - ttimes[0]) * 0.00136986

    for c, t, p, l, u in z_holdout:
        htimes.append(t)
        hlabels.append('set 2')
    #
    # print 'holdout 0', htimes[0]
    # print 'holdout -1', htimes[-1]
    # print (htimes[-1] - htimes[0]) * 0.00136986

    d = loadmat('/mnt/sda3/data/kaggle-seizure-prediction/labels/%s_answer_key.mat' % s.replace('_holdout', ''))
    pre_times_mat = list(d['answer_key']['preictal_uutc'][0][0][0] * 2.77778e-10)
    int_times_mat = list(d['answer_key']['interictal_uutc'][0][0][0] * 2.77778e-10)
    train_times = pre_times_mat + int_times_mat
    train_labels = ['train'] * len(train_times)

    train_diff = max(train_times) - min(train_times)

    patch_handles = []
    left = 0

    # set 1
    patch_handles.append(ax.barh(i, ttimes[-1] - ttimes[0], height=0.3,
                                 color=c_test, align='center',
                                 left=left, edgecolor="none", label='test' if i==0 else None ))
    left += ttimes[-1] - ttimes[0]
    patch = patch_handles[-1][0]
    bl = patch.get_xy()
    x = 0.5 * patch.get_width() + bl[0]
    y = 0.5 * patch.get_height() + bl[1]
    nmonths = (max(ttimes) - min(ttimes)) * 0.00136986
    nhours = len(ttimes)
    ax.text(x, y, "%.0fh/%.1fm" % (nhours, nmonths), ha='center', va='center', fontsize=16)

    # set 2
    patch_handles.append(ax.barh(i, htimes[-1] - htimes[0], height=0.3,
                                 color=c_holdout, align='center',
                                 left=left, edgecolor="none", label='hold-out'if i==0 else None ))
    patch = patch_handles[-1][0]
    bl = patch.get_xy()
    x = 0.5 * patch.get_width() + bl[0]
    y = 0.5 * patch.get_height() + bl[1]
    nmonths = (max(htimes) - min(htimes)) * 0.00136986
    nhours = len(htimes)
    ax.text(x, y, "%.0fh/%.1fm" % (nhours, nmonths), ha='center', va='center', fontsize=16)

ax.set_yticks(zip(*zs)[0])
subjects_real = [s.replace('_holdout', '') for s in subjects]
ax.set_yticklabels(subjects_real, fontsize=20)
ax.set_xlabel('Chronological order', fontsize=20)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.09), fancybox=True,
#           ncol=2, fontsize=20)

ax.legend(loc='best', fancybox=True, fontsize=20)


plt.tick_params(
    axis='x',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom='off',  # ticks along the bottom edge are off
    top='off',  # ticks along the top edge are off
    labelbottom='off')
ax.set_xlim([-1, 10462])
plt.tight_layout()
plt.savefig('/mnt/sda3/CODING/python/kaggle-seizure-predict/metadata/images/test-holdout-custom-split.eps', format='eps')
plt.show()
