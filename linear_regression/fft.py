import json
import os
import shutil
import numpy as np
from preprocessors.filtering import run_filter_preprocessor
from scipy.io import savemat, loadmat
from functools import partial
from multiprocessing import Pool
from utils.config_name_creator import *
from pandas import DataFrame


def group_into_bands(fft, fft_freq, nfreq_bands):
    if nfreq_bands == 67:
        bands = range(1, 50, 1) + range(50, 100, 5) + range(100, 181, 10)
    else:
        raise ValueError('wrong nfreq_bands')
    freq_bands = np.digitize(fft_freq, bands)
    df = DataFrame({'fft': fft, 'band': freq_bands})
    df = df.groupby('band').mean()
    return df.fft[1:-1]


def compute_fft(x, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features):
    n_channels = x.shape[0]
    n_timesteps = (data_length_sec - win_length_sec) / stride_sec + 1
    x2 = np.zeros((n_channels, nfreq_bands, n_timesteps))
    for i in range(n_channels):
        xc = np.zeros((nfreq_bands, n_timesteps))
        for frame_num, w in enumerate(range(0, data_length_sec - win_length_sec + 1, stride_sec)):
            xw = x[i, w * sampling_frequency: (w + win_length_sec) * sampling_frequency]
            fft = np.log10(np.absolute(np.fft.rfft(xw)))
            fft_freq = np.fft.rfftfreq(n=xw.shape[-1], d=1.0 / sampling_frequency)
            xc[:nfreq_bands, frame_num] = group_into_bands(fft, fft_freq, nfreq_bands)
        x2[i, :, :] = xc
    return x2


def get_files_paths(directory, extension='.mat'):
    filenames = sorted(os.listdir(directory))
    files_with_extension = [directory + '/' + f for f in filenames if f.endswith(extension) and not f.startswith('.')]
    return files_with_extension


def process_file(read_dir, write_dir, nfreq_bands, win_length_sec, stride_sec, features, raw_file_path):
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


def run_fft_preprocessor():
    with open('SETTINGS.json') as f:
        settings_dict = json.load(f)

    # path
    input_data_path = settings_dict['path']['processed_data_path'] + '/' + create_time_data_name(settings_dict)
    output_data_path = settings_dict['path']['processed_data_path'] + '/' + create_fft_data_name(settings_dict)

    # params
    nfreq_bands = settings_dict['preprocessor']['nfreq_bands']
    win_length_sec = settings_dict['preprocessor']['win_length_sec']
    stride_sec = settings_dict['preprocessor']['stride_sec']
    features = settings_dict['preprocessor']['features']

    if not os.path.exists(input_data_path):
        run_filter_preprocessor()

    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)
    shutil.copy2('SETTINGS.json', output_data_path + '/SETTINGS.json')

    subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    for subject in subjects:
        print '>> fft', subject
        read_dir = input_data_path + '/' + subject
        write_dir = output_data_path + '/' + subject

        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        raw_files = get_files_paths(read_dir)
        pool = Pool(6)
        part_f = partial(process_file, read_dir, write_dir, nfreq_bands,
                         win_length_sec, stride_sec, features)
        pool.map(part_f, raw_files)


if __name__ == '__main__':
    run_fft_preprocessor()

