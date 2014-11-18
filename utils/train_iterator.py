import numpy as np


class RandomTrainIterator(object):
    def __init__(self, dataset, batch_size):
        self.x, self.y = dataset
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
            return x, y
        else:
            self.__restart()
            raise StopIteration

    def __restart(self):
        self.rng.shuffle(self.idx)
        self.batch_index = 0