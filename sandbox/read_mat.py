from scipy.io import loadmat
import utils

subject = 'Dog_4_holdout'
d = loadmat('/mnt/sda3/data/kaggle-seizure-prediction/labels/%s_answer_key.mat' % subject)
hclass_times = list(d['answer_key']['testing_uutc'][0][0][0]* 2.77778e-10)
hclass_labels = list(d['answer_key']['classification_vector'][0][0][0])
hl = ['holdout'] * len(hclass_times)

print min(hclass_times)


subject = 'Dog_1'
d = loadmat('/mnt/sda3/data/kaggle-seizure-prediction/labels/%s_answer_key.mat' % subject)
class_times = list(d['answer_key']['testing_uutc'][0][0][0]* 2.77778e-10)
class_labels = list(d['answer_key']['classification_vector'][0][0][0])
lt = ['test'] * len(class_times)
last_test = sorted(class_times)[-1]

t = hclass_times + class_times
cl = hclass_labels + class_labels
l = hl + lt

z = zip(t,l, cl)
z.sort(key=lambda x: x[0])

for i, (t,l,c) in enumerate(z):
    print i, t,l, c
    if t == last_test :
        print 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'

print len(hclass_times)



class_times.sort()
for i in class_times:
    print i

t_all = t_int + t_pre
filenames = utils.load_pkl('filenames.pkl')
f_int = [f for g in filenames[subject]['interictal'] for f in g]
f_pre = [f for g in filenames[subject]['preictal'] for f in g]
fnames = f_int + f_pre

labels_all = [0] * len(t_int) + [1] * len(t_pre)
zipped = zip(t_all, labels_all, fnames)
zipped.sort(key=lambda x: x[0])

t_all.sort()
diff = [i - j for i, j in zip(t_all[1:], t_all[:-1])]
diff.append(0)

for i, (t, l, f) in enumerate(zipped):
    print t, l, f, diff[i]


print len(fnames), len(t_all)
print len(f_int), len(t_int)
print len(f_pre), len(t_pre)
