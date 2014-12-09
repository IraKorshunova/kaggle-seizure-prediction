import numpy as np
import copy


class FoldIterator(object):
    def __init__(self, dataset):
        self.x_preictal, self.x_interictal = dataset['preictal'], dataset['interictal']
        n_preictal_hours = len(self.x_preictal)
        n_interictal_hours  = len(self.x_interictal)
        n_interictal_blocks = n_interictal_hours/n_preictal_hours

        self.rng = np.random.RandomState(56051315)
        self.idx = np.arange(len(self.x))
        self.batch_size = batch_size
        self.data_size = self.x.shape[0]
        self.__restart()

    def __iter__(self):
        return self

    def next(self):
        begin = self.batch_index
        end = self.batch_index + self.batch_size
        idx_to_read = self.idx[begin:end]
        if end <= self.data_size:
            x = self.x[idx_to_read]
            y = self.y[idx_to_read]
            self.batch_index += self.batch_size
            #TODO: hack here
            # x_shuffled = x.copy()
            # self.rng.shuffle(np.rollaxis(x_shuffled,-1))
            # return x_shuffled, y
            return x, y
        else:
            self.__restart()
            raise StopIteration

    def __restart(self):
        self.rng.shuffle(self.idx)
        self.batch_index = 0
