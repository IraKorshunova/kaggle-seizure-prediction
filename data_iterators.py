import numpy as np
import pathfinder
import preprocess
import loader
import glob
import utils


class DataGenerator(object):
    def __init__(self, subject, dataset, batch_size, transform_params, scalers=None,
                 full_batch=False, random=True, infinite=False, store_in_ram=True):

        data_path = preprocess.preprocess_data(pathfinder.RAW_DATA_PATH, subject, transform_params)
        print data_path
        if store_in_ram:
            if dataset == 'train':
                x, y, filename2idx, idx2filename = loader.load_train_data(data_path, subject)
                self.x, self.scalers = loader.scale_across_time(x)
                self.y = y
            elif dataset == 'test':
                x, filename2idx, idx2filename = loader.load_test_data(data_path, subject)
                assert scalers is not None
                self.x, _ = loader.scale_across_time(x, scalers)
                self.y = np.zeros((len(x),), dtype='int32')  # fake labels
            elif dataset == 'train-valid':
                x, y, filename2idx, idx2filename = loader.load_train_data(data_path, subject, dataset='train',
                                                                          proportion_valid=0.3)
                self.x, self.scalers = loader.scale_across_time(x)
                self.y = y
            elif dataset == 'valid':
                x, y, filename2idx, idx2filename = loader.load_train_data(data_path, subject, dataset='valid',
                                                                          proportion_valid=0.3)
                self.y = y
                assert scalers is not None
                self.x, _ = loader.scale_across_time(x, scalers)
            elif dataset == 'holdout':
                x, filename2idx, idx2filename = loader.load_holdout_data(data_path, subject)
                self.y = np.zeros((len(x),), dtype='int32')
                assert scalers is not None
                self.x, _ = loader.scale_across_time(x, scalers)

            self.filename2idx = filename2idx
            self.idx2filename = idx2filename

            self.n_channels = x.shape[1]
            self.n_fbins = x.shape[2]
            self.n_timesteps = x.shape[3]
            self.nsamples = x.shape[0]
        else:
            if dataset == 'train':
                self.data_paths = glob.glob(data_path + '/%s/*ictal_segment_*.mat' % subject)
            else:
                self.data_paths = glob.glob(data_path + '/%s/*_test_segment_*.mat' % subject)

            x, y, id = loader.read_file(self.data_paths[0])
            self.x = np.expand_dims(x, axis=0)
            self.n_channels = self.x.shape[1]
            self.n_fbins = self.x.shape[2]
            self.n_timesteps = self.x.shape[3]
            self.nsamples = len(self.data_paths)

        self.batch_size = batch_size
        self.rng = np.random.RandomState(42)
        self.full_batch = full_batch
        self.random = random
        self.infinite = infinite
        self.transformation_params = transform_params
        self.ram = store_in_ram

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                nb = len(idxs_batch)
                x_batch = np.zeros((nb,) + self.x.shape[1:], dtype='float32')
                y_batch = np.zeros((nb,), dtype='int32')
                ids_batch = []

                for i, j in enumerate(idxs_batch):
                    if self.ram:
                        x_batch[i] = self.x[j]
                        y_batch[i] = self.y[j]
                        ids_batch.append(self.idx2filename[j])
                    else:
                        x, y, id = loader.read_file(self.data_paths[j])
                        x_batch[i] = x
                        y_batch[i] = y
                        ids_batch.append(id)

                if self.full_batch:
                    if nb == self.batch_size:
                        yield x_batch, y_batch, ids_batch
                else:
                    yield x_batch, y_batch, ids_batch
            if not self.infinite:
                break


class TrainAugmentDataGenerator(object):
    def __init__(self, subject, batch_size, transform_params, scalers=None,
                 full_batch=False, random=True, infinite=True):

        data_path = preprocess.preprocess_data(pathfinder.RAW_DATA_PATH, subject, transform_params)
        print data_path
        x, y, filename2idx, idx2filename = loader.load_train_data(data_path, subject)
        self.x, self.scalers = loader.scale_across_time(x)
        self.y = y

        self.filename2idx = filename2idx
        self.idx2filename = idx2filename

        self.n_channels = x.shape[1]
        self.n_fbins = x.shape[2]
        self.n_timesteps = x.shape[3]
        self.nsamples = x.shape[0]

        self.batch_size = batch_size
        self.rng = np.random.RandomState(42)
        self.full_batch = full_batch
        self.random = random
        self.infinite = infinite
        self.transformation_params = transform_params

        d = utils.load_pkl('filenames.pkl')
        grouped_filenames = d[subject]['preictal'] + d[subject]['interictal']
        new_grouped_filenames = []
        for g in grouped_filenames:
            new_g = []
            for f in g:
                new_f = f.split('/')[-1]
                new_g.append(new_f)
            new_grouped_filenames.append(new_g)
        self.grouped_filenames = new_grouped_filenames
        self.neighbors = {}

    def find_neighbors(self, filename):
        if filename not in self.neighbors:
            for group in self.grouped_filenames:
                for j, f in enumerate(group):
                    if f == filename:
                        prev = group[j - 1] if j > 0 else None
                        next = group[j + 1] if j < len(group) - 1 else None
                        self.neighbors[filename] = (prev, next)
                        return prev, next
        else:
            return self.neighbors[filename]

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                nb = len(idxs_batch)
                x_batch = np.zeros((nb,) + self.x.shape[1:], dtype='float32')
                y_batch = np.zeros((nb,), dtype='int32')
                ids_batch = []

                for i, j in enumerate(idxs_batch):
                    if self.rng.randint(0, 2):
                        fname = self.idx2filename[j]
                        prev_fname, next_fname = self.find_neighbors(fname)
                        prev_neighbor = 0
                        if prev_fname and next_fname:
                            prev_neighbor = self.rng.randint(0, 2)
                        if prev_fname and not next_fname:
                            prev_neighbor = 1
                        if next_fname and not prev_fname:
                            prev_neighbor = 0

                        if prev_neighbor:
                            prev_idx = self.filename2idx[prev_fname]
                            offset = self.rng.randint(1, self.n_timesteps + 1)
                            x_batch[i] = np.concatenate((self.x[prev_idx, :, :, offset:],
                                                         self.x[j, :, :, :offset]), axis=-1)
                        else:
                            next_idx = self.filename2idx[next_fname]
                            offset = self.rng.randint(1, self.n_timesteps + 1)
                            x_batch[i] = np.concatenate((self.x[j, :, :, offset:],
                                                         self.x[next_idx, :, :, :offset]), axis=-1)

                    else:
                        x_batch[i] = self.x[j]

                    y_batch[i] = self.y[j]
                    ids_batch.append(self.idx2filename[j])

                if self.full_batch:
                    if nb == self.batch_size:
                        yield x_batch, y_batch, ids_batch
                else:
                    yield x_batch, y_batch, ids_batch
            if not self.infinite:
                break


class HoursDataGenerator(object):
    def __init__(self, subject, batch_size, transform_params, dataset='train', scalers=None,
                 full_batch=False, random=True, infinite=True, groups_x=None, groups_y=None, groups_f=None):

        data_path = preprocess.preprocess_data(pathfinder.RAW_DATA_PATH, subject, transform_params)
        if dataset == 'train':
            if groups_x is None and groups_y is None and groups_f is None:
                train_groups_x, train_groups_y, train_groups_filenames, _ = loader.load_grouped_train_data(data_path,
                                                                                                           subject)
            else:
                train_groups_x, train_groups_y, train_groups_filenames = groups_x, groups_y, groups_f
            x, y, ids = [], [], []
            for gx, gy, gid in zip(train_groups_x, train_groups_y, train_groups_filenames):
                if len(gx) == 6:
                    x_matrix = np.concatenate(gx, axis=-1)
                    x.append(x_matrix)
                    y.append(gy[0])
                    ids.append(gid[0])
                else:
                    print 'Skip sequence of length', len(gx)

            x = np.stack(x, axis=0)
            y = np.stack(y, axis=0)
            self.x, self.scalers = loader.scale_across_time(x)
            self.y = y
            self.ids = ids

        elif dataset == 'test':
            test_clip2label, test_clip2time, test_clip2usage = loader.load_test_labels(pathfinder.LABELS_PATH)
            test_preictal_groups, test_interictal_groups = loader.group_labels_by_hour(
                test_clip2label, test_clip2time,
                subject)
            if groups_x is None and groups_y is None and groups_f is None:
                test_groups_x, test_groups_y, test_groups_filenames = loader.load_grouped_test_data(data_path,
                                                                                                    test_preictal_groups,
                                                                                                    test_interictal_groups,
                                                                                                    subject)
            else:
                test_groups_x, test_groups_y, test_groups_filenames = groups_x, groups_y, groups_f
            x, y, ids = [], [], []
            for gx, gy, gid in zip(test_groups_x, test_groups_y, test_groups_filenames):
                if len(gx) == 6:
                    x_matrix = np.concatenate(gx, axis=-1)
                    x.append(x_matrix)
                    y.append(gy[0])
                    ids.append(gid[0])
            x = np.stack(x, axis=0)
            y = np.stack(y, axis=0)

            assert scalers is not None
            self.x, _ = loader.scale_across_time(x, scalers)
            self.y = y
            self.ids = ids

        elif dataset == 'holdout':
            holdout_clip2label, holdout_clip2time = loader.load_holdout_labels(pathfinder.LABELS_PATH)
            holdout_preictal_groups, holdout_interictal_groups = loader.group_labels_by_hour(
                holdout_clip2label, holdout_clip2time, subject)

            holdout_groups_x, holdout_groups_y, holdout_groups_filenames = loader.load_grouped_test_data(data_path,
                                                                                                         holdout_preictal_groups,
                                                                                                         holdout_interictal_groups,
                                                                                                         subject)
            x, y, ids = [], [], []
            for gx, gy, gid in zip(holdout_groups_x, holdout_groups_y, holdout_groups_filenames):
                if len(gx) == 6:
                    x_matrix = np.concatenate(gx, axis=-1)
                    x.append(x_matrix)
                    y.append(gy[0])
                    ids.append(gid[0])
            x = np.stack(x, axis=0)
            y = np.stack(y, axis=0)

            assert scalers is not None
            self.x, _ = loader.scale_across_time(x, scalers)
            self.y = y
            self.ids = ids
        elif dataset == 'set1':
            set1, set2 = timesplit_test_holdout(data_path, subject)
            set1_groups_x, set1_groups_y, set1_groups_filenames, _, _ = zip(*set1)

            train_groups_x, train_groups_y, train_groups_filenames, _ = loader.load_grouped_train_data(data_path,
                                                                                                       subject)
            train_groups_x.extend(set1_groups_x)
            train_groups_y.extend(set1_groups_y)
            train_groups_filenames.extend(set1_groups_filenames)

            x = []
            for gx in train_groups_x:
                if len(gx) == 6:
                    x_matrix = np.concatenate(gx, axis=-1)
                    x.append(x_matrix)
                else:
                    print 'Skip sequence of length', len(gx)

            x = np.stack(x, axis=0)
            _, self.scalers = loader.scale_across_time(x)

            x, y, ids = [], [], []
            for gx, gy, gid in zip(set1_groups_x, set1_groups_y, set1_groups_filenames):
                if len(gx) == 6:
                    x_matrix = np.concatenate(gx, axis=-1)
                    x.append(x_matrix)
                    y.append(gy[0])
                    ids.append(gid[0])
                else:
                    print 'Skip sequence of length', len(gx)

            x = np.stack(x, axis=0)
            y = np.stack(y, axis=0)
            self.x, _ = loader.scale_across_time(x, self.scalers)
            self.y = y
            self.ids = ids

        elif dataset == 'set2':
            set1, set2 = timesplit_test_holdout(data_path, subject)
            set2_groups_x, set2_groups_y, set2_groups_filenames, _, _ = zip(*set2)
            x, y, ids = [], [], []
            for gx, gy, gid in zip(set2_groups_x, set2_groups_y, set2_groups_filenames):
                if len(gx) == 6:
                    x_matrix = np.concatenate(gx, axis=-1)
                    x.append(x_matrix)
                    y.append(gy[0])
                    ids.append(gid[0])
            x = np.stack(x, axis=0)
            y = np.stack(y, axis=0)

            assert scalers is not None
            self.x, _ = loader.scale_across_time(x, scalers)
            self.y = y
            self.ids = ids
        elif dataset == 'train_test':
            train_groups_x, train_groups_y, train_groups_filenames, _ = loader.load_grouped_train_data(data_path,
                                                                                                           subject)
            x, y, ids = [], [], []

            for gx, gy, gid in zip(train_groups_x, train_groups_y, train_groups_filenames):
                if len(gx) == 6:
                    x_matrix = np.concatenate(gx, axis=-1)
                    x.append(x_matrix)
                    y.append(gy[0])
                    ids.append(gid[0])
                else:
                    print 'Skip sequence of length', len(gx)


            test_clip2label, test_clip2time, test_clip2usage = loader.load_test_labels(pathfinder.LABELS_PATH)
            test_preictal_groups, test_interictal_groups = loader.group_labels_by_hour(
                test_clip2label, test_clip2time,
                subject)
            test_groups_x, test_groups_y, test_groups_filenames = loader.load_grouped_test_data(data_path,
                                                                                                    test_preictal_groups,
                                                                                                    test_interictal_groups,
                                                                                                    subject)
            for gx, gy, gid in zip(test_groups_x, test_groups_y, test_groups_filenames):
                if len(gx) == 6:
                    x_matrix = np.concatenate(gx, axis=-1)
                    x.append(x_matrix)
                    y.append(gy[0])
                    ids.append(gid[0])

            x = np.stack(x, axis=0)
            y = np.stack(y, axis=0)
            self.x, self.scalers = loader.scale_across_time(x)
            self.y = y
            self.ids = ids
        else:
            raise ValueError()

        self.n_channels = x.shape[1]
        self.n_fbins = x.shape[2]
        self.n_timesteps = x.shape[3]
        self.nsamples = x.shape[0]

        self.batch_size = batch_size
        self.rng = np.random.RandomState(42)
        self.full_batch = full_batch
        self.random = random
        self.infinite = infinite
        self.transformation_params = transform_params

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                nb = len(idxs_batch)
                x_batch = np.zeros((nb,) + self.x.shape[1:], dtype='float32')
                y_batch = np.zeros((nb,), dtype='int32')
                ids_batch = []

                for i, j in enumerate(idxs_batch):
                    x_batch[i] = self.x[j]
                    y_batch[i] = self.y[j]
                    ids_batch.append(self.ids[j])

                if self.full_batch:
                    if nb == self.batch_size:
                        yield x_batch, y_batch, ids_batch
                else:
                    yield x_batch, y_batch, ids_batch
            if not self.infinite:
                break


def timesplit_test_holdout(data_path, subject):
    test_clip2label, test_clip2time, test_clip2usage = loader.load_test_labels(pathfinder.LABELS_PATH)
    test_preictal_groups, test_interictal_groups = loader.group_labels_by_hour(test_clip2label, test_clip2time, subject)
    test_groups_x, test_groups_y, test_groups_filenames = loader.load_grouped_test_data(data_path,
                                                                                        test_preictal_groups,
                                                                                        test_interictal_groups,
                                                                                        subject)
    print len(test_groups_x)

    holdout_clip2label, holdout_clip2time = loader.load_holdout_labels(pathfinder.LABELS_PATH)
    holdout_preictal_groups, holdout_interictal_groups = loader.group_labels_by_hour(
        holdout_clip2label, holdout_clip2time, subject)

    holdout_groups_x, holdout_groups_y, holdout_groups_filenames = loader.load_grouped_test_data(data_path,
                                                                                                 holdout_preictal_groups,
                                                                                                 holdout_interictal_groups,
                                                                                            subject + '_holdout')
    print len(holdout_groups_x)
    groups_x = test_groups_x + holdout_groups_x
    groups_y = test_groups_y + holdout_groups_y
    groups_filenames = test_groups_filenames + holdout_groups_filenames
    group_times = [test_clip2time[t[0]] for t in test_groups_filenames] + \
                  [holdout_clip2time[t[0]] for t in holdout_groups_filenames]
    usage = ['test'] * len(test_groups_filenames) + ['holdout'] * len(holdout_groups_filenames)

    z = zip(groups_x, groups_y, groups_filenames, group_times, usage)
    z.sort(key=lambda x: x[3])
    last_test_idx = 0
    for i, (_, _, f, t, u) in enumerate(z):
        # print f, t, u
        if u == 'test':
            last_test_idx = i

    half = last_test_idx if 'Dog_1' in subject or 'Dog_4' in subject else len(z) / 2
    print len(z[:half]), len(z[half:])
    return z[:half], z[half:]
