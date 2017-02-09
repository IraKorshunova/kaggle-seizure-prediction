import numpy as np
import matplotlib.pyplot as plt

thresholds_lda_calibr = (0.70014599, 0.49628664, 0.53560483, 0.53153214, 0.42603192, 0.7180398, 0.67409046)
thresholds_lda = (0.28541213, 0.09444624, 0.3900553, 0.0929177, 1.43579939e-38, 0.99864978, 0.7105083)
subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
clda_c = '#7fcdbb'
clda = '#2c7fb8'

ind = np.array([0, 1.1, 2.2, 3.3, 4.4, 5.8, 7.3])  # the x locations for the groups
width = 0.4  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, thresholds_lda, width, color=clda)
rects2 = ax.bar(ind + width, thresholds_lda_calibr, width, color=clda_c)

# add some text for labels, title and axes ticks
ax.set_ylabel('Optimal threshold', fontsize=20)
ax.set_xticks(ind + width)
ax.set_xticklabels(subjects, fontsize=18)
ax.legend((rects1[0], rects2[0]), ('LDA', 'LDA calibrated'),
          loc='upper left', fancybox=True, fontsize=20)

for label in ax.get_yticklabels():
        label.set_fontsize(18)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                '%.2f' % height,
                ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
ax.set_xlim([-0.1, 8.2])
ax.set_ylim([0., 1.05])

plt.tight_layout()
plt.savefig('../metadata/images/thresholds.eps', format='eps')
plt.show()
