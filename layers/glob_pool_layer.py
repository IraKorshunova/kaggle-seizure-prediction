from numpy.core.multiarray import dtype
import theano
import theano.tensor as T
import numpy as np


class GlobalPoolLayer(object):
    def __init__(self, input):
        input += 1e-06
        avg = input.mean(3, dtype='float32')
        max = input.max(3)
        min = input.min(3)
        var = input.var(3)
        geom_mean = T.exp(T.mean(T.log(input), axis=3, dtype='float32'))
        l2_norm = input.norm(2, axis=3)
        self.output = T.concatenate([avg, max, min, var, geom_mean, l2_norm], axis=2)