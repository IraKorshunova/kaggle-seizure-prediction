import utils
from scipy.io import loadmat
import loader
import numpy as np
import csv

# data_path = '/mnt/sda3/data/kaggle-seizure-prediction/labels'
# subject = 'Dog_5'
#
# csv_filepath = data_path + '/answer.csv'
# clip2label, clip2usage = {}, {}
# clips_csv, labels_csv = [], []
# with open(csv_filepath, 'rb') as f:
#     reader = csv.reader(f)
#     reader.next()
#     for row in reader:
#         clip = row[0]
#         label = int(row[1])
#         usage = row[2]
#         clip2label[clip] = label
#         clip2usage[clip] = usage
#         if subject in clip:
#             clips_csv.append(clip)
#             label = -1 if label == 0 else 1
#             labels_csv.append(label)
#
# clips2label, _, _ = loader.load_test_labels(data_path)
# s_clip2label = utils.get_subject_records(clips2label, subject)
#
# d = loadmat(data_path + '/' + subject + '_answer_key.mat')
# order = d['answer_key']['testing_array'][0][0][0][:-1]
# times = d['answer_key']['testing_uutc'][0][0][0] * 2.77778e-10
# labels = d['answer_key']['classification_vector'][0][0][0]
#
# assert np.equal(labels_csv, labels).all()
#
# zipped = zip(times, order, labels, labels_csv, clips_csv)
# zipped.sort(key=lambda x: x[1])
# for t, o, l, lc, id, in zipped:
#     print id, t, o, l, lc




##############################################################
##############################################################
##############################################################
##############################################################
##############################################################
#
# data_path = '/mnt/sda3/data/kaggle-seizure-prediction/labels'
# subject = 'Dog_1_holdout'
#
# csv_filepath = data_path + '/holdout_key.csv'
# clip2label = {}
# clips_csv, labels_csv = [], []
# with open(csv_filepath, 'rb') as f:
#     reader = csv.reader(f)
#     reader.next()
#     for row in reader:
#         clip = row[0]
#         label = int(row[1])
#         clip2label[clip] = label
#         if subject in clip:
#             clips_csv.append(clip)
#             labels_csv.append(label)
#
# clips2label, clip2time2 = loader.load_holdout_labels(data_path)
# s_clip2label = utils.get_subject_records(clips2label, subject)
#
# d = loadmat(data_path + '/' + subject + '_answer_key.mat')
# order = d['answer_key']['testing_array'][0][0][0][:-1]
# times_mat = d['answer_key']['testing_uutc'][0][0][0] * 2.77778e-10
# labels_mat = d['answer_key']['classification_vector'][0][0][0]
#
# lll = [-1 if l == 0 else 1 for l in labels_csv]
# assert np.equal(lll, labels_mat).all()
#
# # zipped = zip(times, order, labels, labels_csv, clips_csv)
# # zipped.sort(key=lambda x: x[0])
# # stimes = [i[0] for i in zipped]
# # diffs = np.array(stimes)[1:] - np.array(stimes)[:-1]
# # for i, (t, o, l, lc, id) in enumerate(zipped):
# #     print id, t, o, l, lc, diffs[i]
#
# clip2time = {}
# for l_mat, l_csv, c_csv, t_mat in zip(labels_mat, labels_csv, clips_csv, times_mat):
#     l = 0 if l_mat == -1 else 1
#     assert l == l_csv
#
#     clip2time[c_csv] = t_mat * 2.77778e-10  # convert to hours
#
# preictal_groups, interictal_groups = loader.group_labels_by_hour(clip2label, clip2time, subject)
#
# print preictal_groups
# for g in interictal_groups:
#     for clip in g:
#         assert clip2time[clip] == clip2time2[clip]
#         print clip, clip2time[clip]

###########################################################
###########################################################
###########################################################
###########################################################
###########################################################

data_path = '/mnt/sda3/data/kaggle-seizure-prediction/labels'
subject = 'Dog_1'
clips2label = loader.load_train_labels(data_path)
s_clip2label = utils.get_subject_records(clips2label, subject)
n_pre, n_int = 0, 0
for label in s_clip2label.itervalues():
    if label == 0:
        n_int += 1
    else:
        n_pre += 1

d = loadmat(data_path + '/' + subject + '_answer_key.mat')
pre_times_mat = list(d['answer_key']['preictal_uutc'][0][0][0] * 2.77778e-10)
int_times_mat = list(d['answer_key']['interictal_uutc'][0][0][0] * 2.77778e-10)
pre_filenames = [utils.create_filename(subject, s, 'preictal') for s in range(1, len(pre_times_mat) + 1)]
int_filenames = [utils.create_filename(subject, s, 'interictal') for s in range(1, len(int_times_mat)+1)]
pre_clip2time = dict([(c, t) for t, c in zip(pre_times_mat, pre_filenames)])
times = pre_times_mat + int_times_mat
filenames = pre_filenames + int_filenames

print len(int_times_mat), n_int
print len(pre_times_mat), n_pre

clip2time = {}
for f, t in zip(filenames, times):
    clip2time[f] = t

filenames2 = utils.load_pkl('filenames.pkl')[subject]
g_int = filenames2['interictal']
g_pre = filenames2['preictal']

groups, times = [], []
for g in g_int:
    group_times = [clip2time[f.split('/')[-1]] for f in g]
    diffs = np.array(group_times)[1:] - np.array(group_times)[:-1]
    # print g , diffs
    groups.append(g)
    times.append(group_times)


for g in g_pre:
    group_times = [clip2time[f.split('/')[-1]] for f in g]
    diffs = np.array(group_times)[1:] - np.array(group_times)[:-1]
    # print g , diffs
    groups.append(g)
    times.append(group_times)

print len(groups)
print len(times)
z = zip(groups, times)
z.sort(key=lambda x:max(x[1]))
print len(groups)


for g,t in z:
    print g,t



#
# zipped = zip(times, filenames)
# zipped.sort(key=lambda x: x[0])
# stimes = [i[0] for i in zipped]
# diffs = np.array(stimes)[1:] - np.array(stimes)[:-1]
# for i, (t, f) in enumerate(zipped):
#     print f, t, diffs[i]
