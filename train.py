import cPickle as pickle
import string
import sys
from itertools import izip
import lasagne as nn
import numpy as np
import theano
import utils
import logger
from configuration import config, set_configuration
import pathfinder
import data_iterators

nn.random.set_rng(np.random.RandomState(42))
np.random.seed(42)


def train(subject):
    metadata_path = metadata_exp_dir + '/' + subject + '.pkl'

    print 'Load data'
    if config().transformation_params.get('augment'):
        train_data_iterator = data_iterators.TrainAugmentDataGenerator(subject=subject, batch_size=config().batch_size,
                                                                       transform_params=config().transformation_params,
                                                                       full_batch=True, random=True, infinite=True)
        print 'Augmentation'
    elif config().transformation_params.get('hours'):
        train_data_iterator = data_iterators.HoursDataGenerator(subject=subject, dataset='train',
                                                                batch_size=config().batch_size,
                                                                transform_params=config().transformation_params,
                                                                full_batch=True, random=True, infinite=True)
        print 'Train on hour sequences'

    else:
        train_data_iterator = data_iterators.DataGenerator(subject=subject, dataset='train',
                                                           batch_size=config().batch_size,
                                                           transform_params=config().transformation_params,
                                                           full_batch=True, random=True, infinite=True)

    nbatch_per_epoch = int(train_data_iterator.nsamples / config().batch_size)
    if hasattr(config(), 'max_nbatch'):
        max_nbatch = config().max_nbatch
    else:
        max_nbatch = config().max_epochs * nbatch_per_epoch

    n_channels = train_data_iterator.n_channels
    n_fbins = train_data_iterator.n_fbins
    n_timesteps = train_data_iterator.n_timesteps
    print
    print 'Data'
    print 'n train: %d' % train_data_iterator.nsamples

    print 'Build model'
    model = config().build_model(n_channels, n_fbins, n_timesteps)
    all_layers = nn.layers.get_all_layers(model.l_out)
    num_params = nn.layers.count_params(model.l_out)
    print '  number of parameters: %d' % num_params
    print string.ljust('  layer output shapes:', 36),
    print string.ljust('#params:', 10),
    print 'output shape:'
    for layer in all_layers:
        name = string.ljust(layer.__class__.__name__, 32)
        num_param = sum([np.prod(p.get_value().shape) for p in layer.get_params()])
        num_param = string.ljust(num_param.__str__(), 10)
        print '    %s %s %s' % (name, num_param, layer.output_shape)

    train_loss = config().build_objective(model)

    learning_rate_schedule = config().learning_rate_schedule
    learning_rate = theano.shared(np.float32(learning_rate_schedule[0]))
    updates = config().build_updates(train_loss, model, learning_rate)

    xs_shared = nn.utils.shared_empty(dim=len(model.l_in.shape), dtype='float32')
    ys_shared = nn.utils.shared_empty(dim=len(model.l_targets.shape), dtype='int32')

    givens_train = {}
    givens_train[model.l_in.input_var] = xs_shared
    givens_train[model.l_targets.input_var] = ys_shared

    iter_train = theano.function([], train_loss, givens=givens_train, updates=updates,
                                 on_unused_input='ignore')
    batch_idxs = range(max_nbatch)
    losses_train = []

    print
    print 'Train model for %s iterations' % max_nbatch
    tmp_losses_train = []

    for batch_idx, (xs_batch, ys_batch, _) in izip(batch_idxs, train_data_iterator.generate()):
        epoch = 1. * batch_idx / nbatch_per_epoch
        if epoch in learning_rate_schedule:
            lr = np.float32(learning_rate_schedule[epoch])
            print '  setting learning rate to %.7f' % lr
            print
            learning_rate.set_value(lr)

        xs_shared.set_value(xs_batch)
        ys_shared.set_value(ys_batch)
        loss = iter_train()
        tmp_losses_train.append(loss)

        if ((epoch + 1) % config().save_every) == 0:
            print 'Epoch:', epoch
            print 'Mean training loss', np.mean(tmp_losses_train)
            losses_train.append(np.mean(tmp_losses_train))
            tmp_losses_train = []

            print 'Saving metadata, parameters'

            with open(metadata_path, 'w') as f:
                pickle.dump({
                    'configuration': config_name,
                    'git_revision_hash': utils.get_git_revision_hash(),
                    'experiment_id': expid,
                    'batch_since_start': batch_idx,
                    'losses_train': losses_train,
                    'param_values': nn.layers.get_all_param_values(model.l_out)
                }, f, pickle.HIGHEST_PROTOCOL)

                print '  saved to %s' % metadata_path
                print


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit("Usage: train.py <configuration_name>")

    config_name = sys.argv[1]
    set_configuration(config_name)
    expid = utils.generate_expid(config_name)
    print
    print "Experiment ID: %s" % expid
    print

    # metadata
    metadata_dir = utils.get_dir_path('train', pathfinder.METADATA_PATH)
    metadata_exp_dir = metadata_dir + '/%s' % expid
    utils.make_dir(metadata_exp_dir)

    # logs
    logs_dir = utils.get_dir_path('logs', pathfinder.METADATA_PATH)
    sys.stdout = logger.Logger(logs_dir + '/%s.log' % expid)
    sys.stderr = sys.stdout

    if len(sys.argv) == 3:
        subjects = [sys.argv[2]]
    else:
        subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    for subject in subjects:
        print '***********************', subject, '***************************'
        # try:
        #     train(subject)
        # except:
        #     pass
        train(subject)
