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
max_epochs = 5000
save_every = 10

batch_size = 32
learning_rate_schedule = {0: 0.03, 100: 0.01, 200: 0.003, 300: 0.001, 4000: 0.0001}


def build_model(n_channels, n_fbins, n_timesteps):
    l_in = nn.layers.InputLayer((None, n_channels, n_fbins, n_timesteps))
    lr = nn.layers.ReshapeLayer(l_in, ([0], n_channels * n_fbins, n_timesteps))
    lr = nn.layers.DimshuffleLayer(lr, (0, 2, 1))

    input_gate = nn.layers.Gate(W_in=nn.init.GlorotUniform(), W_hid=nn.init.Orthogonal())
    forget_gate = nn.layers.Gate(W_in=nn.init.GlorotUniform(), W_hid=nn.init.Orthogonal(), b=nn.init.Constant(5.0))
    output_gate = nn.layers.Gate(W_in=nn.init.GlorotUniform(), W_hid=nn.init.Orthogonal())
    cell = nn.layers.Gate(W_in=nn.init.GlorotUniform(), W_hid=nn.init.Orthogonal(), W_cell=None,
                          nonlinearity=nn.nonlinearities.tanh)

    l1 = nn.layers.LSTMLayer(lr, num_units=128,
                             hid_init=nn.init.Orthogonal(),
                             ingate=input_gate, forgetgate=forget_gate,
                             cell=cell, outgate=output_gate,
                             peepholes=False,
                             precompute_input=False,
                             grad_clipping=5)

    input_gate = nn.layers.Gate(W_in=nn.init.GlorotUniform(), W_hid=nn.init.Orthogonal())
    forget_gate = nn.layers.Gate(W_in=nn.init.GlorotUniform(), W_hid=nn.init.Orthogonal(), b=nn.init.Constant(5.0))
    output_gate = nn.layers.Gate(W_in=nn.init.GlorotUniform(), W_hid=nn.init.Orthogonal())
    cell = nn.layers.Gate(W_in=nn.init.GlorotUniform(), W_hid=nn.init.Orthogonal(), W_cell=None,
                          nonlinearity=nn.nonlinearities.tanh)
    l2 = nn.layers.LSTMLayer(l1, num_units=128,
                             hid_init=nn.init.Orthogonal(),
                             ingate=input_gate, forgetgate=forget_gate,
                             cell=cell, outgate=output_gate,
                             peepholes=False,
                             precompute_input=False,
                             grad_clipping=5)

    l_rshp = nn.layers.ReshapeLayer(l2, (-1, 128))

    l_out_sm1 = nn.layers.DenseLayer(l_rshp, num_units=2,
                                     nonlinearity=nn.nonlinearities.softmax)

    l_out = nn.layers.ReshapeLayer(l_out_sm1, (-1, n_timesteps, 2))
    l_out_sm = nn.layers.SliceLayer(l_out, indices=-1, axis=1)
    l_out = nn.layers.SliceLayer(l_out_sm, indices=1, axis=1)

    l_targets = nn.layers.InputLayer((None,), input_var=T.ivector('tgt'))

    return namedtuple('Model', ['l_in', 'l_out', 'l_targets', 'l_out_sm', 'l_out_sm1']) \
        (l_in, l_out, l_targets, l_out_sm, l_out_sm1)


def build_objective(model, deterministic=False):
    predictions = nn.layers.get_output(model.l_out_sm, deterministic=deterministic)
    targets = nn.layers.get_output(model.l_targets)
    loss = nn.objectives.categorical_crossentropy(predictions, targets)

    predictions_sm1 = nn.layers.get_output(model.l_out_sm1, deterministic=deterministic)
    targets_rshp = T.flatten(T.outer(targets, T.ones((10,), dtype='int32')))
    loss_sm1 = nn.objectives.categorical_crossentropy(predictions_sm1, targets_rshp)

    layers = {l: l2_reg for l in nn.layers.get_all_layers(model.l_out)}
    l2_penalty = nn.regularization.regularize_layer_params_weighted(layers, nn.regularization.l2)

    return T.mean(loss) + T.mean(loss_sm1) + l2_penalty


def build_updates(train_loss, model, learning_rate):
    updates = nn.updates.adam(train_loss, nn.layers.get_all_params(model.l_out), learning_rate)
    return updates
