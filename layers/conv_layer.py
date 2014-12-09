import numpy as np
import theano
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import theano.tensor as T


class ConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize, activation, weights_variance, subsample):

        assert image_shape[1] == filter_shape[1]
        self.input =input

        if activation == 'tanh':
            activation_function = lambda x: T.tanh(x)
            fan_in = np.prod(filter_shape[1:])
            fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            W_values = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype='float32')
            b_values = np.zeros((filter_shape[0],), dtype='float32')

        elif activation == 'relu':
            activation_function = lambda x: T.maximum(0.0, x)
            W_values = np.asarray(rng.normal(0.0, weights_variance, size=filter_shape), dtype='float32')
            b_values = np.ones((filter_shape[0],), dtype='float32') / 10.0
        else:
            raise ValueError('unknown activation function')

        self.W = theano.shared(value=W_values, name='W', borrow=True)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        conv_out = conv.conv2d(input, self.W, filter_shape=filter_shape, image_shape=image_shape, subsample=subsample)
        pooled_out = downsample.max_pool_2d(conv_out, poolsize, ignore_border=True) if poolsize[1] > 1 else conv_out
        self.output = activation_function(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.weights = [self.W, self.b]