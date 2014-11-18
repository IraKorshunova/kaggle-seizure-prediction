import random
import numpy as np
import itertools, copy


def split_data_with_overlap(data_grouped_by_hour, valid_size, overlap_size, window_size,
                            overlap_interictal=True, overlap_preictal=True, random_state=42):
    random.seed(random_state)

    number_of_test_interictal_hours = max(1, int(len(data_grouped_by_hour['interictal']) * valid_size))
    number_of_test_preictal_hours = max(1, int(len(data_grouped_by_hour['preictal']) * valid_size))

    interictal_hours_indexes = range(len(data_grouped_by_hour['interictal']))
    preictal_hours_indexes = range(len(data_grouped_by_hour['preictal']))

    valid_interictal_hours_indexes = random.sample(interictal_hours_indexes, number_of_test_interictal_hours)
    valid_preictal_hours_indexes = random.sample(preictal_hours_indexes, number_of_test_preictal_hours)

    train_interictal_hours_indexes = [idx for idx in interictal_hours_indexes if
                                      idx not in set(valid_interictal_hours_indexes)]
    train_preictal_hours_indexes = [idx for idx in preictal_hours_indexes if
                                    idx not in set(valid_preictal_hours_indexes)]

    def fill_data_list(class_label, indexes):
        overlap = overlap_preictal if class_label == 'preictal' else overlap_interictal
        x = []
        for idx in indexes:
            data_hour = data_grouped_by_hour[class_label][idx]
            if overlap:
                data = np.concatenate(data_hour, axis=2)
                for i in xrange(divmod(data.shape[-1] - overlap_size, window_size - overlap_size)[0]):
                    i *= window_size - overlap_size
                    x.append(data[..., i:i + window_size])
            else:
                x.extend(data_hour)
        return x

    x_valid_interictal = fill_data_list('interictal', valid_interictal_hours_indexes)
    x_valid_preictal = fill_data_list('preictal', valid_preictal_hours_indexes)
    x_train_interictal = fill_data_list('interictal', train_interictal_hours_indexes)
    x_train_preictal = fill_data_list('preictal', train_preictal_hours_indexes)

    x_valid = x_valid_interictal + x_valid_preictal
    y_valid = len(x_valid_interictal) * [0] + len(x_valid_preictal) * [1]
    combined = zip(x_valid, y_valid)
    random.shuffle(combined)
    x_valid[:], y_valid[:] = zip(*combined)

    x_train = x_train_interictal + x_train_preictal
    y_train = len(x_train_interictal) * [0] + len(x_train_preictal) * [1]
    combined = zip(x_train, y_train)
    random.shuffle(combined)
    x_train[:], y_train[:] = zip(*combined)

    return np.array(x_train, dtype='float32'), np.array(y_train, dtype='int8'), \
           np.array(x_valid, dtype='float32'), np.array(y_valid, dtype='int8')


def generate_overlapped_data(data_grouped_by_hour, overlap_size, window_size,
                                       overlap_interictal=True, overlap_preictal=True, random_state=42):
    random.seed(random_state)

    interictal_hours_indexes = range(len(data_grouped_by_hour['interictal']))
    preictal_hours_indexes = range(len(data_grouped_by_hour['preictal']))

    def fill_data_list(class_label, indexes):
        overlap = overlap_preictal if class_label == 'preictal' else overlap_interictal
        x = []
        for idx in indexes:
            data_hour = data_grouped_by_hour[class_label][idx]
            if overlap:
                data = np.concatenate(data_hour, axis=2)
                for i in xrange(divmod(data.shape[-1] - overlap_size, window_size - overlap_size)[0]):
                    i *= window_size - overlap_size
                    x.append(data[..., i:i + window_size])
            else:
                x.extend(data_hour)
        return x

    x_interictal = fill_data_list('interictal', interictal_hours_indexes)
    x_preictal = fill_data_list('preictal', preictal_hours_indexes)

    x = x_interictal + x_preictal
    y = len(x_interictal) * [0] + len(x_preictal) * [1]
    combined = zip(x, y)
    random.shuffle(combined)
    x[:], y[:] = zip(*combined)

    return np.array(x, dtype='float32'), np.array(y, dtype='int8')


def split_train_valid_filenames(subject, filenames_grouped_by_hour, random_state=42):
    rng = np.random.RandomState(random_state)

    preictal = copy.deepcopy(filenames_grouped_by_hour[subject]['preictal'])
    rng.shuffle(preictal)
    interictal = copy.deepcopy(filenames_grouped_by_hour[subject]['interictal'])
    rng.shuffle(interictal)

    n_preictal = len(preictal)
    n_interictal = len(interictal)

    n_valid_preictal = int(max(1, np.round(0.25 * n_preictal)))
    n_valid_interictal = int(max(1, np.round(0.25 * n_interictal)))

    valid = preictal[:n_valid_preictal] + interictal[:n_valid_interictal]
    valid = list(itertools.chain.from_iterable(valid))

    train = preictal[n_valid_preictal:] + interictal[n_valid_interictal:]
    train = list(itertools.chain.from_iterable(train))

    return {'train_filenames': train, 'valid_filnames': valid}

