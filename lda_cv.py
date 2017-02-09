import sys
import numpy as np
import utils
from configuration import config, set_configuration
import pathfinder
import preprocess
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import loader
from sklearn.cross_validation import StratifiedKFold


def cv(subject):
    print 'Load data'
    data_path = preprocess.preprocess_data(pathfinder.RAW_DATA_PATH, subject, config().transformation_params)
    groups_x, groups_y, groups_filenames, groups_t = loader.load_grouped_train_data(data_path, subject, full_hour=True)
    print len(groups_x)
    z = zip(groups_x, groups_y, groups_filenames, groups_t)
    z.sort(key=lambda x: max(x[3]))
    groups_x, groups_y, groups_filenames, groups_t = zip(*z)
    kfoldy = [g[0] for g in groups_y]
    cv_predictions, cv_labels = [], []
    kf = StratifiedKFold(kfoldy, n_folds=4)
    for train_idx, test_idx in kf:
        # print 'train-test idx', train_idx, test_idx
        train_groups_x = [groups_x[t] for t in train_idx]
        train_groups_y = [groups_y[t] for t in train_idx]
        x = np.stack([e for group in train_groups_x for e in group])
        y_10m = np.stack([e for group in train_groups_y for e in group])
        print 'n_train_groups', len(train_groups_x)

        x, y = utils.reshape_data_for_lda(x, y_10m)
        data_scaler = StandardScaler()
        x = data_scaler.fit_transform(x)

        model = LinearDiscriminantAnalysis()
        model.fit(x, y)

        for t in test_idx:
            x_test = np.stack(groups_x[t])
            y_test = np.stack(groups_y[t])
            x_test = utils.reshape_data_for_lda(x_test)
            x_test = data_scaler.transform(x_test)
            try:
                pred_1m = model.predict_proba(x_test)[:, 1]
                pred_1h = np.mean(pred_1m, axis=0)
                cv_predictions.append(pred_1h)
                cv_labels.append(y_test[0])
            except:
                print 'failed'

    predictions_path = predictions_exp_dir + '/' + subject + '-cv.pkl'
    d = {'predictions': cv_predictions, 'targets': cv_labels}
    utils.save_pkl(d, predictions_path)
    print 'Saved to', predictions_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit("Usage: train.py <configuration_name>")

    config_name = sys.argv[1]
    set_configuration(config_name)
    expid = utils.generate_expid(config_name)
    print
    print "Experiment ID: %s" % expid
    print

    # predictions paths
    prediction_dir = utils.get_dir_path('predictions', pathfinder.METADATA_PATH)
    predictions_exp_dir = utils.find_model_metadata(prediction_dir, config_name)

    if len(sys.argv) == 3:
        subjects = [sys.argv[2]]
    else:
        subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4']

    for subject in subjects:
        print '***********************', subject, '***************************'
        cv(subject)
