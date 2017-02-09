import sys
import numpy as np
import theano
import lasagne as nn
import utils
from configuration import config, set_configuration
import pathfinder
import data_iterators
import string
import glob
import csv
import kaggle_auc


def predict_train_test(subject):
    predictions_path_train = predictions_exp_dir + '/' + subject + '-train.pkl'
    predictions_path_test = predictions_exp_dir + '/' + subject + '-test.pkl'

    metadata_path = metadata_exp_dir + '/' + subject + '.pkl'
    metadata = utils.load_pkl(metadata_path)

    print 'Load data'
    if config().transformation_params.get('hours'):
        train_data_iterator = data_iterators.HoursDataGenerator(subject=subject, dataset='train',
                                                                batch_size=config().batch_size,
                                                                transform_params=config().transformation_params,
                                                                full_batch=False, random=False, infinite=False)
        test_data_iterator = data_iterators.HoursDataGenerator(subject=subject, dataset='test',
                                                               batch_size=config().batch_size,
                                                               transform_params=config().transformation_params,
                                                               scalers=train_data_iterator.scalers,
                                                               full_batch=False, random=False, infinite=False)
    else:
        train_data_iterator = data_iterators.DataGenerator(subject=subject, batch_size=config().batch_size,
                                                           dataset='train',
                                                           transform_params=config().transformation_params,
                                                           full_batch=False, random=False, infinite=False)

        test_data_iterator = data_iterators.DataGenerator(subject=subject, batch_size=config().batch_size,
                                                          dataset='test',
                                                          transform_params=config().transformation_params,
                                                          scalers=train_data_iterator.scalers,
                                                          full_batch=False, random=False, infinite=False)

    n_channels = train_data_iterator.n_channels
    n_fbins = train_data_iterator.n_fbins
    n_timesteps = train_data_iterator.n_timesteps

    print
    print 'Data'
    print 'n train: %d' % train_data_iterator.nsamples
    print 'n test: %d' % test_data_iterator.nsamples

    print 'Build model'
    model = config().build_model(n_channels, n_fbins, n_timesteps)
    nn.layers.set_all_param_values(model.l_out, metadata['param_values'])

    xs_shared = nn.utils.shared_empty(dim=len(model.l_in.shape), dtype='float32')
    givens_in = {model.l_in.input_var: xs_shared}
    iter_test_det = theano.function([], nn.layers.get_output(model.l_out, deterministic=True), givens=givens_in)

    # ------- TRAIN
    batch_predictions, batch_targets, batch_ids = [], [], []
    for xs_batch, ys_batch, ids_batch in train_data_iterator.generate():
        xs_shared.set_value(xs_batch)
        batch_predictions.append(iter_test_det())
        batch_targets.append(ys_batch)
        batch_ids.extend(ids_batch)

    targets, predictions = np.concatenate(batch_targets, axis=0), np.concatenate(batch_predictions, axis=0)
    print '\n Train AUC:', kaggle_auc.auc(targets, predictions)
    print '\n Train loss:', utils.cross_entropy_loss(targets, predictions)
    id2prediction = {}
    for i in xrange(len(batch_ids)):
        id2prediction[batch_ids[i]] = predictions[i]
    utils.save_pkl(id2prediction, predictions_path_train)
    print ' predictions saved to %s' % predictions_path_train

    # ------- TEST
    batch_predictions, batch_ids = [], []
    for xs_batch, _, ids_batch in test_data_iterator.generate():
        xs_shared.set_value(xs_batch)
        batch_predictions.append(iter_test_det())
        batch_ids.extend(ids_batch)

    predictions = np.concatenate(batch_predictions, axis=0)
    id2prediction = {}
    for i in xrange(len(batch_ids)):
        id2prediction[batch_ids[i]] = predictions[i]
    utils.save_pkl(id2prediction, predictions_path_test)
    print ' predictions saved to %s' % predictions_path_test


def predict_holdout(subject):
    predictions_path_holdout = predictions_exp_dir + '/' + subject + '.pkl'
    subject_real = subject.replace('_holdout', '')

    metadata_path = metadata_exp_dir + '/' + subject_real + '.pkl'
    metadata = utils.load_pkl(metadata_path)

    print 'Load data'
    if config().transformation_params.get('hours'):
        train_data_iterator = data_iterators.HoursDataGenerator(subject=subject_real, dataset='train',
                                                                batch_size=config().batch_size,
                                                                transform_params=config().transformation_params,
                                                                full_batch=False, random=False, infinite=False)
        holdout_data_iterator = data_iterators.HoursDataGenerator(subject=subject, dataset='holdout',
                                                                  batch_size=config().batch_size,
                                                                  transform_params=config().transformation_params,
                                                                  scalers=train_data_iterator.scalers,
                                                                  full_batch=False, random=False, infinite=False)
    else:
        train_data_iterator = data_iterators.DataGenerator(subject=subject_real,
                                                           batch_size=config().batch_size,
                                                           dataset='train',
                                                           transform_params=config().transformation_params,
                                                           full_batch=False, random=False, infinite=False)

        holdout_data_iterator = data_iterators.DataGenerator(subject=subject, batch_size=config().batch_size,
                                                             dataset='holdout',
                                                             transform_params=config().transformation_params,
                                                             scalers=train_data_iterator.scalers,
                                                             full_batch=False, random=False, infinite=False)
    n_channels = train_data_iterator.n_channels
    n_fbins = train_data_iterator.n_fbins
    n_timesteps = train_data_iterator.n_timesteps

    print
    print 'Data'
    print 'n holdout: %d' % holdout_data_iterator.nsamples

    print 'Build model'
    model = config().build_model(n_channels, n_fbins, n_timesteps)
    nn.layers.set_all_param_values(model.l_out, metadata['param_values'])

    xs_shared = nn.utils.shared_empty(dim=len(model.l_in.shape), dtype='float32')
    givens_in = {model.l_in.input_var: xs_shared}
    iter_test_det = theano.function([], nn.layers.get_output(model.l_out, deterministic=True), givens=givens_in)

    # ------- HOLDOUT
    batch_predictions, batch_ids = [], []
    for xs_batch, _, ids_batch in holdout_data_iterator.generate():
        xs_shared.set_value(xs_batch)
        batch_predictions.append(iter_test_det())
        batch_ids.extend(ids_batch)

    predictions = np.concatenate(batch_predictions, axis=0)
    id2prediction = {}
    for i in xrange(len(batch_ids)):
        id2prediction[batch_ids[i]] = predictions[i]
    utils.save_pkl(id2prediction, predictions_path_holdout)
    print ' predictions saved to %s' % predictions_path_holdout


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit("Usage: predict.py <config_name>")

    config_name = sys.argv[1]

    metadata_dir = utils.get_dir_path('train', pathfinder.METADATA_PATH)
    metadata_exp_dir = utils.find_model_metadata(metadata_dir, config_name)
    metadata_files = glob.glob(metadata_exp_dir + '/*.pkl')
    metadata = utils.load_pkl(metadata_files[0])  # pick random metadata file

    assert config_name == metadata['configuration']
    set_configuration(config_name)

    # predictions paths
    prediction_dir = utils.get_dir_path('predictions', pathfinder.METADATA_PATH)
    predictions_exp_dir = prediction_dir + "/%s" % metadata['experiment_id']
    utils.make_dir(predictions_exp_dir)

    # submissions paths
    submission_dir = utils.get_dir_path('submissions', pathfinder.METADATA_PATH)
    submission_path_test = submission_dir + "/%s.csv" % metadata['experiment_id']
    submission_path_holdout = submission_dir + "/holdout-%s.csv" % metadata['experiment_id']

    subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    for subject in subjects:
        predict_train_test(subject)

    utils.make_submission_file(predictions_exp_dir, submission_path_test, subjects)

    holdout_subjects = ['Dog_1_holdout', 'Dog_2_holdout', 'Dog_3_holdout', 'Dog_4_holdout']
    for subject in holdout_subjects:
        predict_holdout(subject)
