import lasagne as nn
import theano.tensor as T
from collections import namedtuple
import numpy as np

try:
    from lasagne.layers.dnn import Conv2DDNNLayer
except:
    pass
restart_from_save = None
transformation_params = {
    'highcut': 180,
    'lowcut': 0.1,
    'nfreq_bands': 8,
    'win_length_sec': 60,
    'features': 'meanlog',
    'stride_sec': 60,
}

l2_reg = 0.0001
max_epochs = 7000
save_every = 10

batch_size = 32
learning_rate_schedule = {0: 0.03, 100: 0.01, 200: 0.003, 300: 0.001, 4000: 0.0001}


def build_model(n_channels, n_fbins, n_timesteps):
    alpha = 1./3
    l_in = nn.layers.InputLayer((None, n_channels, n_fbins, n_timesteps))
    lr = nn.layers.ReshapeLayer(l_in, ([0], 1, n_channels * n_fbins, n_timesteps))

    l1c = Conv2DDNNLayer(lr, num_filters=16, filter_size=(n_channels * n_fbins, 1),
                         W=nn.init.Orthogonal(np.sqrt(2 / (1 + alpha ** 2))),
                         b=nn.init.Constant(0.1),
                         nonlinearity=nn.nonlinearities.very_leaky_rectify)
    l2c = Conv2DDNNLayer(l1c, num_filters=32, filter_size=(1, 1),
                         W=nn.init.Orthogonal(np.sqrt(2 / (1 + alpha ** 2))),
                         b=nn.init.Constant(0.1),
                         nonlinearity=nn.nonlinearities.very_leaky_rectify)

    lgp_mean = nn.layers.GlobalPoolLayer(l2c, pool_function=T.mean)
    lgp_max = nn.layers.GlobalPoolLayer(l2c, pool_function=T.max)
    lgp_min = nn.layers.GlobalPoolLayer(l2c, pool_function=T.min)
    lgp_var = nn.layers.GlobalPoolLayer(l2c, pool_function=T.var)

    lgp = nn.layers.ConcatLayer([lgp_mean, lgp_max, lgp_min, lgp_var])

    ld = nn.layers.DenseLayer(nn.layers.dropout(lgp), 512, W=nn.init.Orthogonal(),
                              nonlinearity=nn.nonlinearities.tanh)
    l_out_sm = nn.layers.DenseLayer(nn.layers.dropout(ld), 2, nonlinearity=nn.nonlinearities.softmax,
                                    W=nn.init.Orthogonal())
    l_out = nn.layers.SliceLayer(l_out_sm, indices=1, axis=1)

    l_targets = nn.layers.InputLayer((None,), input_var=T.ivector('tgt'))

    reg_layers = [ld, l_out_sm]

    return namedtuple('Model', ['l_in', 'l_out', 'l_targets', 'l_out_sm', 'reg_layers'])(l_in, l_out, l_targets,
                                                                                         l_out_sm, reg_layers)


def build_objective(model, deterministic=False):
    predictions = nn.layers.get_output(model.l_out_sm, deterministic=deterministic)
    targets = nn.layers.get_output(model.l_targets)
    loss = nn.objectives.categorical_crossentropy(predictions, targets)

    layers = {l: l2_reg for l in nn.layers.get_all_layers(model.l_out)}
    l2_penalty = nn.regularization.regularize_layer_params_weighted(layers, nn.regularization.l2)

    return T.mean(loss) + l2_penalty


def build_updates(train_loss, model, learning_rate):
    updates = nn.updates.adam(train_loss, nn.layers.get_all_params(model.l_out), learning_rate)
    return updates
