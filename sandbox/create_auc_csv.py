import utils
import loader
import csv

cnn_path_test = '/mnt/sda3/CODING/python/kaggle-seizure-predict/metadata/predictions/n8b1m_relu_sm-schaap-20160523-234959'
lda_path_test = '/mnt/sda3/CODING/python/kaggle-seizure-predict/metadata/predictions/lda_8b1m-schaap-20160519-110913'
svm_path_test = '/mnt/sda3/CODING/python/kaggle-seizure-predict/metadata/predictions/svm_rbf_c10-paard-20160607-163326'

clip2label,_ = loader.load_holdout_labels('/mnt/sda3/data/kaggle-seizure-prediction/labels')
clip2label_test, _,_ = loader.load_test_labels('/mnt/sda3/data/kaggle-seizure-prediction/labels')
clip2label.update(clip2label_test)

clip2pred_cnn, clip2pred_lda, clip2pred_svm = {}, {}, {}

subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
for s in subjects:
    prediction_file = cnn_path_test + '/%s-test.pkl' % s
    clip2pred_cnn.update(utils.load_pkl(prediction_file))

    prediction_file = lda_path_test + '/%s-test.pkl' % s
    clip2pred_lda.update(utils.load_pkl(prediction_file))

    prediction_file = svm_path_test + '/%s-test.pkl' % s
    clip2pred_svm.update(utils.load_pkl(prediction_file))

subjects = ['Dog_2_holdout', 'Dog_3_holdout']
for s in subjects:
    prediction_file = cnn_path_test + '/%s.pkl' % s
    clip2pred_cnn.update(utils.load_pkl(prediction_file))

    prediction_file = lda_path_test + '/%s.pkl' % s
    clip2pred_lda.update(utils.load_pkl(prediction_file))

    prediction_file = svm_path_test + '/%s.pkl' % s
    clip2pred_svm.update(utils.load_pkl(prediction_file))

with open('aaa.csv', 'wb') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['clip', 'cnn', 'svm', 'lda', 'label'])
    for k in sorted(clip2pred_cnn.keys()):
        csv_writer.writerow(
            [k, str(clip2pred_cnn[k]), str(clip2pred_svm[k]), str(clip2pred_lda[k]), str(clip2label[k])])
