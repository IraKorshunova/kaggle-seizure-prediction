import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, training_mode, dropout_prob, activation, weights_variance):
        self.input = input
        if activation == 'tanh':
            activation_function = lambda x: T.tanh(x)
            W_values = np.asarray(rng.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype='float32')
            b_values = np.zeros((n_out,), dtype='float32')

        elif activation == 'relu':
            activation_function = lambda x: T.maximum(0.0, x)
            W_values = np.asarray(rng.normal(0.0, weights_variance, size=(n_in, n_out)), dtype='float32')
            b_values = np.ones((n_out,), dtype='float32') / 10.0
        else:
            raise ValueError('unknown activation function')

        self.W = theano.shared(value=W_values, name='W', borrow=True)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        inv_dropout_prob = np.float32(1.0 - dropout_prob)
        lin_output = ifelse(T.eq(training_mode, 1),
                            T.dot(self._dropout(rng, input, dropout_prob), self.W) + self.b,
                            T.dot(input, inv_dropout_prob * self.W) + self.b)

        self.output = activation_function(lin_output)
        self.weights = [self.W, self.b]

    def _dropout(self, rng, layer, p):
        srng = T.shared_randomstreams.RandomStreams(rng.randint(777777))
        mask = srng.binomial(n=1, p=1 - p, size=layer.shape)
        output = layer * T.cast(mask, 'float32')
        return output
