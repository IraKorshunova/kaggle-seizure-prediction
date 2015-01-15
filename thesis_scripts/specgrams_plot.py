import json
import os
import shutil
import numpy as np
from preprocessors.filtering import run_filter_preprocessor
from scipy.io import savemat, loadmat
from functools import partial
from multiprocessing import Pool
from test_labels_loader.config_name_creator import *
from pandas import DataFrame
import matplotlib.pyplot as plt


def group_into_bands(fft, fft_freq, nfreq_bands):
    if nfreq_bands == 67:
        bands = range(1, 50, 1) + range(50, 100, 5) + range(100, 181, 10)
    elif nfreq_bands == 4:
        bands = [0.1, 4, 8, 12, 30]
    elif nfreq_bands == 6:
        bands = [0.1, 4, 8, 12, 30, 70, 180]
    # http://onlinelibrary.wiley.com/doi/10.1111/j.1528-1167.2011.03138.x/pdf
    elif nfreq_bands == 8:
        bands = [0.1, 4, 8, 12, 30, 50, 70, 100, 180]
    else:
        raise ValueError('wrong nfreq_bands')
    freq_bands = np.digitize(fft_freq, bands)
    df = DataFrame({'fft': fft, 'band': freq_bands})
    df = df.groupby('band').mean()
    return df.fft[1:-1]


def compute_fft(x, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features):
    #nfreq_bands=12001
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
            #xc[:,frame_num] = fft
        x2[i, :, :] = xc
    return x2


def get_files_paths(directory, extension='.mat'):
    filenames = sorted(os.listdir(directory))
    files_with_extension = [directory + '/' + f for f in filenames if f.endswith(extension) and not f.startswith('.')]
    return files_with_extension


def process_file(nfreq_bands, win_length_sec, stride_sec, features, raw_file_path):

    d = loadmat(raw_file_path, squeeze_me=True)
    x = d['data']
    data_length_sec = d['data_length_sec']
    sampling_frequency = d['sampling_frequency']

    new_x = compute_fft(x, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features)

    ax = plt.gca()
    ax.set_yticks(range(0,6))
    ax.set_yticklabels([ 'delta', 'theta', 'alpha',
                         'beta', 'low-gamma', 'high-gamma'])
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)
    ax.set_xticks(range(0,10))
    ax.set_xticklabels(range(0,10))
    plt.imshow(new_x[0, :, :], aspect='auto', origin='lower', interpolation='none')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    plt.xlabel('Time, min', fontsize=20)
    plt.show()

    ax = plt.gca()
    bands = np.arange(0, 181, 30)* x.shape[-1]/(sampling_frequency)/10
    ax.set_yticks(bands)
    ax.set_yticklabels(np.arange(0, 181, 30))
    ax.set_xticks(range(0,10))
    ax.set_xticklabels(range(0,10))
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(new_x[0, :, :], aspect='auto', origin='lower', interpolation='none')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    plt.xlabel('Time, min', fontsize=20)
    plt.ylabel('Frequency, Hz', fontsize=20)
    plt.show()

def run_fft_preprocessor():
    with open('SETTINGS.json') as f:
        settings_dict = json.load(f)

    # path
    input_data_path = settings_dict['path']['processed_data_path'] + '/' + create_time_data_name(settings_dict)

    # params
    nfreq_bands = settings_dict['preprocessor']['nfreq_bands']
    win_length_sec = settings_dict['preprocessor']['win_length_sec']
    stride_sec = settings_dict['preprocessor']['stride_sec']
    features = settings_dict['preprocessor']['features']

    subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    for subject in subjects:
        read_dir = input_data_path + '/' + subject
        raw_files = get_files_paths(read_dir)
        print raw_files[0]
        process_file(nfreq_bands, win_length_sec, stride_sec, features, raw_files[0])


if __name__ == '__main__':
    run_fft_preprocessor()

