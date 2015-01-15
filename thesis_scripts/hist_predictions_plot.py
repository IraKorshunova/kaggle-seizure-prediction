import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv


s1 = '/mnt/sda4/CODING/python/kaggle_data_old/models/nfreq6featumeanlog_stdhighc180lowcu0.1win_l60strid60globa1recep[1, 2]use_t0activ[urelu, urelu, utanh]dropo[0.3, 0.6]overl9strid[1, 2]train10weigh0.01scale1nkern[32, 64, 512]pool_[1, 1]l2_re0.0001valid10max_i150000rando1/submission'
s2 = '/mnt/sda4/CODING/python/kaggle_data_old/submissions/linear_models/LDA_fft_meanlog_lowcut0.1highcut180nfreq_bands6win_length_sec60stride_sec30'
subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']

for subject in subjects:
    df1 = read_csv(s1 + '/' + subject + '.csv')
    proba1 = df1['preictal']
    df2 = read_csv(s2 + '/' + subject + '.csv')
    proba2 = df2['preictal']
    plt.hist(proba1, color='blue', bins=np.linspace(0, 1, 100), alpha=0.5, label='Convnet', normed=True)
    plt.hist(proba2, color='green', bins=np.linspace(0, 1, 100), alpha=0.5, label='LDA',normed=True)
    plt.legend(loc='upper right')
    plt.xlabel('Preictal probability', fontsize=20)
    plt.ylabel('Relative frequency', fontsize=20)
    ax = plt.gca()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)
    plt.show()