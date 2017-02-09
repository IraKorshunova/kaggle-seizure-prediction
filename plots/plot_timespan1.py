from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import csv
import loader

subjects = ['Dog_4', 'Dog_3', 'Dog_2', 'Dog_1']
fig = plt.figure(figsize=(10, 3.3))
ax = fig.add_subplot(111)
c_private = '#ff3333'
c_holdout = '#bebada'
c_train = '#8dd3c7'
c_public = '#ff3333'

dh = {'Dog_1': 208 + 208, 'Dog_2': 165 + 166, 'Dog_3': 158 + 159, 'Dog_4': 165 * 2}
dm = {'Dog_1': 8.3, 'Dog_2': 4.6, 'Dog_3': 8.1, 'Dog_4': 14.1}
zs = zip(np.arange(0, 4.0, 0.5), subjects)
for i, s in zs:
    print '===========================', s
    try:
        hsubject = s + '_holdout'
        d = loadmat('/mnt/sda3/data/kaggle-seizure-prediction/labels/%s_answer_key.mat' % hsubject)
        htimes = list(d['answer_key']['testing_uutc'][0][0][0] * 2.77778e-10)
        hlabels = ['holdout'] * len(htimes)
    except:
        htimes = []
        hlabels = []

    print 'holdout'
    print len(hlabels)

    clip2label, clip2time, clip2usage = loader.load_test_labels('/mnt/sda3/data/kaggle-seizure-prediction/labels')
    ttimes, tlabels = [], []
    for c in clip2label.iterkeys():
        if s in c:
            ttimes.append(clip2time[c])
            tlabels.append(clip2usage[c])

    print 'test_public'
    npre, nint = 0, 0
    for c, u in clip2usage.iteritems():
        if u == 'Public' and s in c:
            if clip2label[c] == 1:
                npre += 1
            else:
                nint += 1
    print '%s/%s' % (nint, npre)

    print 'test_private'
    npre, nint = 0, 0
    for c, u in clip2usage.iteritems():
        if u == 'Private' and s in c:
            if clip2label[c] == 1:
                npre += 1
            else:
                nint += 1
    print '%s/%s' % (nint, npre)

    d = loadmat('/mnt/sda3/data/kaggle-seizure-prediction/labels/%s_answer_key.mat' % s)
    # ttimes = list(d['answer_key']['testing_uutc'][0][0][0] * 2.77778e-10)
    # tlabels = ['Private'] * len(ttimes)

    pre_times_mat = list(d['answer_key']['preictal_uutc'][0][0][0] * 2.77778e-10)
    int_times_mat = list(d['answer_key']['interictal_uutc'][0][0][0] * 2.77778e-10)

    csv_filepath = '/mnt/sda3/data/kaggle-seizure-prediction/labels/answer-train.csv'
    npre, nint = 0, 0
    with open(csv_filepath, 'rb') as f:
        reader = csv.reader(f)
        reader.next()
        for row in reader:
            clip = row[0]
            label = int(row[1])
            if s in clip:
                if label == 1:
                    npre += 1
                else:
                    nint += 1
    # print npre, nint
    # print len(pre_times_mat), len(int_times_mat)

    train_times = pre_times_mat[:npre] + int_times_mat[:nint]
    train_labels = ['train'] * len(train_times)

    # t = htimes + ttimes + train_times
    # l = hlabels + tlabels + train_labels

    t = htimes + ttimes
    l = hlabels + tlabels
    months = (max(t) - min(t)) * 0.00136986
    hours = len(l) / 6

    z = zip(t, l)
    z.sort(key=lambda x: x[0])
    # for tt, ll in z:
    #     print tt, ll

    t, l = zip(*z)
    diffs = np.array(t)[1:] - np.array(t)[:-1]
    diffs = list([d + 0.166666901961 if d == 0.0 else d for d in diffs])
    diffs.insert(0, 0.0)

    patch_handles = []
    left = 0

    # train
    patch_handles.append(ax.barh(i, len(train_times) * 0.166666901961, height=0.3,
                                 color=c_train, align='center',
                                 left=left, label='train' if i == 0 else None))
    left += len(train_times) * 0.166666901961 + 1
    patch = patch_handles[-1][0]
    bl = patch.get_xy()
    x = 0.5 * patch.get_width() + bl[0]
    y = 0.5 * patch.get_height() + bl[1]
    nmonths = (max(train_times) - min(train_times)) * 0.00136986
    nhours = len(train_times) / 6
    ax.text(x, y, "%.1fm" % (nmonths), ha='center', va='center', fontsize=15)
    # ax.text(x, y, "%.0fh/%.1fm" % (nhours, nmonths), ha='center', va='center', fontsize=15)

    # test
    if 'Dog_4' in s:
        left -= 1
        patch_handles.append(ax.barh(i, len(ttimes) * 0.166666901961, height=0.3,
                                     color=c_public, align='center',
                                     left=left, label='test' if i == 0 else None))
        left += len(ttimes) * 0.166666901961
        patch = patch_handles[-1][0]
        bl = patch.get_xy()
        x = 0.5 * patch.get_width() + bl[0]
        y = 0.5 * patch.get_height() + bl[1]
        nmonths = (max(ttimes) - min(ttimes)) * 0.00136986
        nhours = len(ttimes) / 6
        # ax.text(x, y, "%.0fh/%.1fm" % (nhours, nmonths), ha='center', va='center', fontsize=15)
        ax.text(x, y, "%.1fm" % (nmonths), ha='center', va='center', fontsize=15)

        patch_handles.append(ax.barh(i, len(htimes) * 0.166666901961, height=0.3,
                                     color=c_holdout, align='center',
                                     left=left, label='hold-out' if i == 0 else None))

        left += len(htimes) * 0.166666901961
        patch = patch_handles[-1][0]
        bl = patch.get_xy()
        x = 0.5 * patch.get_width() + bl[0]
        y = 0.5 * patch.get_height() + bl[1]
        nmonths = (max(htimes) - max(min(htimes), max(ttimes))) * 0.00136986
        nhours = len(htimes) / 6
        # ax.text(x, y, "%.0fh/%.1fm" % (nhours, nmonths), ha='center', va='center', fontsize=16)
        ax.text(x, y, "%.1fm" % (nmonths), ha='center', va='center', fontsize=16)

    else:

        for d, ll in zip(diffs, l):
            c = None
            if ll == 'Private':
                c = c_private
            elif ll == 'Public':
                c = c_public
            elif ll == 'holdout':
                c = c_holdout
            elif ll == 'train':
                c = c_train
            else:
                c = None

            if d < 0.5:
                patch_handles.append(ax.barh(i, d, height=0.3,
                                             color=c, align='center',
                                             left=left, edgecolor="none"))
                left += d
            else:
                pass
        if i == 0:
            patch_handles.append(ax.barh(i, 0, height=0.3,
                                         color=c_public, align='center',
                                         left=left, edgecolor="none", label='test'))
            patch_handles.append(ax.barh(i, 0, height=0.3,
                                         color=c_holdout, align='center',
                                         left=left, edgecolor="none", label='hold-out'))

        patch = patch_handles[-1][0]
        bl = patch.get_xy()
        x = patch.get_width() + bl[0] + 2
        y = 0.5 * patch.get_height() + bl[1]
        # ax.text(x, y, "%.0fh/%.1fm" % (hours, months), ha='left', va='center', fontsize=16)

ax.set_yticks(zip(*zs)[0])
ax.set_yticklabels(subjects, fontsize=20)
ax.set_xlabel('Chronological order', fontsize=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), fancybox=True,
          ncol=4, fontsize=16)

plt.tick_params(
    axis='x',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom='off',  # ticks along the bottom edge are off
    top='off',  # ticks along the top edge are off
    labelbottom='off')
ax.set_xlim([-1, 580])
plt.tight_layout()
plt.savefig('/mnt/sda3/CODING/python/kaggle-seizure-predict/metadata/images/data_split.eps', format='eps')
plt.show()
