import numpy as np

import theano
import theano.tensor as T
from theano import Param
from sklearn.metrics import roc_curve, auc

from utils.train_iterator import RandomTrainIterator
from layers.softmax_layer import SoftmaxLayer
from layers.feature_extractor import FeatureExtractor


class ConvNet(object):
    def __init__(self, param_dict):

        self.param_dict = param_dict
        self.training_batch_size = param_dict['training_batch_size']
        nkerns = param_dict['nkerns']
        recept_width = param_dict['recept_width']
        pool_width = param_dict['pool_width']
        stride = param_dict['stride']
        dropout_prob = param_dict['dropout_prob']
        weight_decay = param_dict['l2_reg']
        activation = param_dict['activation']
        weights_variance = param_dict['weights_variance']
        n_channels = param_dict['n_channels']
        n_timesteps = param_dict['n_timesteps']
        n_fbins = param_dict['n_fbins']
        global_pooling = param_dict['global_pooling']
        rng = np.random.RandomState(23455)

        self.training_mode = T.iscalar('training_mode')
        self.x = T.tensor4('x')
        self.y = T.bvector('y')
        self.batch_size = theano.shared(self.training_batch_size)

        self.input = self.x.reshape((self.batch_size, 1, n_channels * n_fbins, n_timesteps))

        self.feature_extractor = FeatureExtractor(rng, self.input, nkerns, recept_width, pool_width, stride,
                                                  self.training_mode,
                                                  dropout_prob[0],
                                                  activation, weights_variance, n_channels, n_timesteps, n_fbins,
                                                  global_pooling)

        self.classifier = SoftmaxLayer(rng=rng, input=self.feature_extractor.output, n_in=nkerns[-1],
                                       training_mode=self.training_mode, dropout_prob=dropout_prob[-1])

        self.weights = self.feature_extractor.weights + self.classifier.weights

        # ---------------------- BACKPROP
        self.cost = self.classifier.cross_entropy_cost(self.y)
        self.cost = self.classifier.cross_entropy_cost(self.y)
        L2_sqr = sum((weight ** 2).sum() for weight in self.weights[::2])
        self.grads = T.grad(self.cost + weight_decay * L2_sqr, self.weights)
        self.updates = self.adadelta_updates(self.grads, self.weights)
        # self.updates = self.nesterov_momentum(self.grads, self.weights)

        # --------------------- FUNCTIONS
        self.train_model = theano.function([self.x, self.y, Param(self.training_mode, default=1)],
                                           outputs=self.cost,
                                           updates=self.updates)

        self.validate_model = theano.function([self.x, self.y, Param(self.training_mode, default=0)],
                                              self.cost)

        self.test_model = theano.function([self.x, Param(self.training_mode, default=0)],
                                          self.classifier.p_y_given_x[:, 1])

    def train(self, train_set, max_iter):
        print 'training for', max_iter, 'iterations'
        self.batch_size.set_value(self.training_batch_size)

        train_set_iterator = RandomTrainIterator(train_set, self.training_batch_size)

        done_looping = False
        iter = 0
        while not done_looping:
            for train_x, train_y in train_set_iterator:
                self.train_model(train_x, train_y)
                # if iter % 10 == 0:
                #     self.batch_size.set_value(train_set[0].shape[0])
                #     print self.validate_model(train_set[0], train_set[1])
                #     self.batch_size.set_value(self.training_batch_size)
                if iter > max_iter:
                    done_looping = True
                    break
                iter += 1

    def validate(self, train_set, valid_set, valid_freq, max_iter, fname_out):

        train_set_iterator = RandomTrainIterator(train_set, self.training_batch_size)
        valid_set_size = len(valid_set[1])

        f_out = open(fname_out, 'w')

        # ------------------------------  TRAINING
        epoch = 0
        iter = 0
        best_ce = np.inf
        best_iter_ce = 0
        best_auc = 0
        best_iter_auc = 0
        done_looping = False

        patience = 100000
        patience_increase = 2
        improvement_threshold = 0.995

        while iter < max_iter and not done_looping:
            epoch += 1
            for train_x, train_y in train_set_iterator:
                self.train_model(train_x, train_y)
                iter += 1
                # ------------------------ VALIDATION
                if iter % valid_freq == 0:

                    self.batch_size.set_value(valid_set_size)
                    cost_valid = self.validate_model(valid_set[0], valid_set[1])
                    auc_valid = self.get_auc(valid_set)

                    # print "%4s %7s  %15s  %15s %10s " % (
                    # epoch, iter, auc_valid, cost_valid,
                    #     patience)
                    f_out.write("%s \t %s  \t %s \n" % (
                        iter, auc_valid, cost_valid))
                    self.batch_size.set_value(self.training_batch_size)

                    if cost_valid <= best_ce:
                        if cost_valid < best_ce * improvement_threshold:
                            patience = max(patience, iter * patience_increase)
                        best_iter_ce = iter
                        best_ce = cost_valid

                    if auc_valid >= best_auc:
                        best_iter_auc = iter
                        best_auc = auc_valid

                if patience <= iter:
                    done_looping = True
        print 'best_iter_cost:', best_iter_ce, 'best_cost:', best_ce
        print 'best_iter_auc:', best_iter_auc, 'best_auc:', best_auc
        f_out.close()
        return max(best_iter_ce, best_iter_auc)

    def get_auc(self, data_xy):
        x, y = data_xy[0], data_xy[1]
        p_y_given_x = self.get_test_proba(x)
        fpr, tpr, thresholds = roc_curve(y, p_y_given_x, pos_label=1)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    def get_test_proba(self, x_test):
        self.batch_size.set_value(len(x_test))
        p_y_given_x = self.test_model(x_test)
        return p_y_given_x

    def nesterov_momentum(self, grads, weights, learning_rate=0.001, momentum=0.9):
        updates = []

        for param_i, grad_i in zip(weights, grads):
            mparam_i = theano.shared(np.zeros(param_i.get_value().shape, dtype=theano.config.floatX))
            v = momentum * mparam_i - learning_rate * grad_i
            w = param_i + momentum * v - learning_rate * grad_i
            updates.append((mparam_i, v))
            updates.append((param_i, w))

        return updates

    def adadelta_updates(self, grads, weights, learning_rate=0.01, rho=0.95, epsilon=1e-6):
        accumulators = [theano.shared(np.zeros_like(param_i.get_value())) for param_i in weights]
        delta_accumulators = [theano.shared(np.zeros_like(param_i.get_value())) for param_i in weights]

        updates = []
        for param_i, grad_i, acc_i, acc_delta_i in zip(weights, grads, accumulators, delta_accumulators):
            acc_i_new = rho * acc_i + (1 - rho) * grad_i ** 2
            updates.append((acc_i, acc_i_new))

            update_i = grad_i * T.sqrt(acc_delta_i + epsilon) / T.sqrt(acc_i_new + epsilon)
            updates.append((param_i, param_i - learning_rate * update_i))

            acc_delta_i_new = rho * acc_delta_i + (1 - rho) * update_i ** 2
            updates.append((acc_delta_i, acc_delta_i_new))

        return updates

    def get_state(self):
        state = {}
        state['params'] = self.param_dict
        weights_vals = []
        for p in self.weights:
            weights_vals.append(p.get_value())
        state['weights'] = weights_vals
        return state

    def set_weights(self, weights_vals):
        for i, w in enumerate(weights_vals):
            self.weights[i].set_value(w)