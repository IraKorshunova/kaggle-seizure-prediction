import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse


class SoftmaxLayer(object):
    def __init__(self, rng, input, n_in, training_mode, dropout_prob=0.0):
        self.W = theano.shared(value=np.zeros((n_in, 2), dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=np.zeros((2,), dtype=theano.config.floatX), name='b', borrow=True)
        inv_dropout_prob = np.float32(1.0 - dropout_prob)
        self.p_y_given_x = ifelse(T.eq(training_mode, 1),
                                  T.nnet.softmax(T.dot(self._dropout(rng, input, dropout_prob), self.W) + self.b),
                                  T.nnet.softmax(T.dot(input, inv_dropout_prob * self.W) + self.b))

        self.weights = [self.W, self.b]

    def cross_entropy_cost(self, y):
        return -T.mean(T.log(self.p_y_given_x + 1e-06)[T.arange(y.shape[0]), y])

    def _dropout(self, rng, layer, p):
        srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
        mask = srng.binomial(n=1, p=1 - p, size=layer.shape)
        output = layer * T.cast(mask, 'float32')
        return output