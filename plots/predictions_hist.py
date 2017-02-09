import matplotlib.pyplot as plt
import numpy as np
import utils
import matplotlib

subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
# subjects = ['Dog_5']
p_cnn = '/mnt/sda3/CODING/python/kaggle-seizure-predict/metadata/predictions/n8b1m_relu_sm-schaap-20160523-234959'
p_lda = '/mnt/sda3/CODING/python/kaggle-seizure-predict/metadata/predictions/lda_8b1m-schaap-20160519-110913'
p_svm = '/mnt/sda3/CODING/python/kaggle-seizure-predict/metadata/predictions/svm_rbf_c10-paard-20160607-163326'
c_cnn = '#edf8b1'
c_lda =  '#2c7fb8'
c_svm = '#7fcdbb'


proba1, proba2, proba3 = [], [], []
holdout_subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4']

for subject in subjects:
    clip2predictions_cnn = utils.load_pkl(p_cnn + '/' + subject + '-test.pkl')
    proba1.extend(clip2predictions_cnn.values())
    if subject in holdout_subjects:
        clip2predictions_cnn = utils.load_pkl(p_cnn + '/' + subject + '_holdout.pkl')
        proba1.extend(clip2predictions_cnn.values())

    clip2predictions_lda = utils.load_pkl(p_lda + '/' + subject + '-test.pkl')
    proba2.extend(clip2predictions_lda.values())
    if subject in holdout_subjects:
        clip2predictions_cnn = utils.load_pkl(p_lda + '/' + subject + '_holdout.pkl')
        proba2.extend(clip2predictions_cnn.values())

    clip2predictions_svm = utils.load_pkl(p_svm + '/' + subject + '-test.pkl')
    proba3.extend(clip2predictions_svm.values())
    if subject in holdout_subjects:
        clip2predictions_cnn = utils.load_pkl(p_svm + '/' + subject + '_holdout.pkl')
        proba2.extend(clip2predictions_cnn.values())

weights = np.ones_like(proba1)/len(proba1)
plt.hist(proba2, color=c_lda, bins=np.linspace(0, 1, 70), alpha=1.0, label='LDA', weights=weights)
plt.hist(proba3, color=c_svm, bins=np.linspace(0, 1, 70), alpha=0.8, label='SVM', weights=weights)
plt.hist(proba1, color=c_cnn, bins=np.linspace(0, 1, 70), alpha=0.7, label='CNN', weights=weights)
plt.legend(loc='upper right', fancybox=True, fontsize=18)
plt.yscale('log', nonposy='clip')

plt.xlabel('Preictal probability', fontsize=20)
plt.ylabel('Log frequency', fontsize=20)
ax = plt.gca()

# ax.set_yticks(np.arange(0, 1,0.1))
# ax.set_yticklabels(np.arange(0, 1.2, 0.2), fontsize=18)

# ax.get_yaxis().set_major_formatter(matplotlib.ticker.LogFormatter())
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(20)

plt.tight_layout()
plt.savefig('../metadata/images/predictions-hist.png')
plt.show()
