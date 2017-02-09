from pandas import DataFrame
import scipy as sc
import scipy.signal
import os
import numpy as np
from functools import partial
from multiprocessing import Pool
from scipy.io import loadmat, savemat
import pathfinder
import utils


def group_into_bands(fft, fft_freq, nfreq_bands):
    if nfreq_bands == 178:
        bands = range(1, 180)
    elif nfreq_bands == 4:
        bands = [0.1, 4, 8, 12, 30]
    elif nfreq_bands == 6:
        bands = [0.1, 4, 8, 12, 30, 70, 180]
    # http://onlinelibrary.wiley.com/doi/10.1111/j.1528-1167.2011.03138.x/pdf
    elif nfreq_bands == 8:
        bands = [0.1, 4, 8, 12, 30, 50, 70, 100, 180]
    elif nfreq_bands == 12:
        bands = [0.1, 4, 8, 12, 30, 40, 50, 60, 70, 85, 100, 140, 180]
    elif nfreq_bands == 9:
        bands = [0.1, 4, 8, 12, 21, 30, 50, 70, 100, 180]
    else:
        raise ValueError('wrong number of frequency bands')
    freq_bands = np.digitize(fft_freq, bands)
    df = DataFrame({'fft': fft, 'band': freq_bands})
    df = df.groupby('band').mean()
    return df.fft[1:-1]


def compute_fft(x, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features):
    n_channels = x.shape[0]
    n_timesteps = (data_length_sec - win_length_sec) / stride_sec + 1
    n_fbins = nfreq_bands + 1 if 'std' in features else nfreq_bands

    x2 = np.zeros((n_channels, n_fbins, n_timesteps))
    for i in range(n_channels):
        xc = np.zeros((n_fbins, n_timesteps))
        for frame_num, w in enumerate(range(0, data_length_sec - win_length_sec + 1, stride_sec)):
            xw = x[i, w * sampling_frequency: (w + win_length_sec) * sampling_frequency]
            if 'pib' in features:
                fft = np.absolute(np.fft.rfft(xw)) ** 2
            elif 'log' in features:
                fft = np.log10(np.absolute(np.fft.rfft(xw)))
            else:
                raise NotImplementedError()
            fft_freq = np.fft.rfftfreq(n=xw.shape[-1], d=1.0 / sampling_frequency)
            xc[:nfreq_bands, frame_num] = group_into_bands(fft, fft_freq, nfreq_bands)
            if 'std' in features:
                xc[-1, frame_num] = np.std(xw)
        x2[i, :, :] = xc
    return x2


def get_files_paths(directory, extension='.mat'):
    filenames = sorted(os.listdir(directory))
    files_with_extension = [directory + '/' + f for f in filenames if f.endswith(extension) and not f.startswith('.')]
    return files_with_extension


def process_file_fft(read_dir, write_dir, nfreq_bands, win_length_sec, stride_sec, features, raw_file_path):
    preprocessed_file_path = raw_file_path.replace(read_dir, write_dir)

    d = loadmat(raw_file_path, squeeze_me=True)
    x = d['data']
    data_length_sec = d['data_length_sec']
    sampling_frequency = d['sampling_frequency']
    sequence = d['sequence']

    new_x = compute_fft(x, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features)
    data_dict = {'data': new_x, 'data_length_sec': data_length_sec, 'sampling_frequency': sampling_frequency,
                 'sequence': sequence}
    savemat(preprocessed_file_path, data_dict)


def filter(x, new_sampling_frequency, data_length_sec, lowcut, highcut):
    x1 = scipy.signal.resample(x, new_sampling_frequency * data_length_sec, axis=1)

    nyq = 0.5 * new_sampling_frequency
    b, a = sc.signal.butter(5, np.array([lowcut, highcut]) / nyq, btype='band')
    x_filt = sc.signal.lfilter(b, a, x1, axis=1)
    return np.float32(x_filt)


def process_file_filtering(read_dir, write_dir, lowcut, highcut, raw_file_path):
    preprocessed_file_path = raw_file_path.replace(read_dir, write_dir)

    d = loadmat(raw_file_path)
    sample = ''
    for key in d.keys():
        if 'segment' in key:
            sample = key
            break
    x = np.array(d[sample][0][0][0], dtype='float32')
    data_length_sec = d[sample][0][0][1][0][0]
    if 'test' in raw_file_path or 'holdout' in raw_file_path:
        sequence = np.Inf
    else:
        sequence = d[sample][0][0][4][0][0]

    new_sampling_frequency = 400
    new_x = filter(x, new_sampling_frequency, data_length_sec, lowcut, highcut)
    data_dict = {'data': new_x, 'data_length_sec': data_length_sec,
                 'sampling_frequency': new_sampling_frequency, 'sequence': sequence}
    savemat(preprocessed_file_path, data_dict, do_compression=True)


def run_fft_preprocessor(in_data_path, subject, transform_params):
    out_data_path = pathfinder.PROCESSED_DATA_PATH + '/' + utils.get_fft_data_name(transform_params)

    nfreq_bands = transform_params['nfreq_bands']
    win_length_sec = transform_params['win_length_sec']
    stride_sec = transform_params['stride_sec']
    features = transform_params['features']

    read_dir = in_data_path + '/' + subject
    write_dir = out_data_path + '/' + subject

    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
        print '>> fft', subject
        raw_files = get_files_paths(read_dir)
        pool = Pool(6)
        part_f = partial(process_file_fft, read_dir, write_dir, nfreq_bands,
                         win_length_sec, stride_sec, features)
        pool.map(part_f, raw_files)

    return out_data_path


def run_filter_preprocessor(in_data_path, subject, transform_params):
    out_data_path = pathfinder.PROCESSED_DATA_PATH + '/' + utils.get_filtered_data_name(transform_params)

    highcut = transform_params['highcut']
    lowcut = transform_params['lowcut']

    read_dir = in_data_path + '/' + subject
    write_dir = out_data_path + '/' + subject

    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
        print '>> filtering', subject
        raw_files = get_files_paths(read_dir)
        process_file_filtering(read_dir, write_dir, lowcut, highcut, raw_files[0])
        pool = Pool(6)
        part_f = partial(process_file_filtering, read_dir, write_dir, lowcut, highcut)
        pool.map(part_f, raw_files)

    return out_data_path


def preprocess_data(in_data_path, subject, transform_params):
    filtered_data_path = run_filter_preprocessor(in_data_path, subject, transform_params)
    if transform_params['features'] == 'raw':
        return filtered_data_path
    fft_data_path = run_fft_preprocessor(filtered_data_path, subject, transform_params)
    return fft_data_path
