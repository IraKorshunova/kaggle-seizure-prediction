import scipy as sc
import scipy.signal
import json
import os
import shutil
import numpy as np
from functools import partial
from multiprocessing import Pool
from scipy.io import loadmat, savemat
from utils.config_name_creator import create_time_data_name


def get_files_paths(directory, extension='.mat'):
    filenames = sorted(os.listdir(directory))
    files_with_extension = [directory + '/' + f for f in filenames if f.endswith(extension) and not f.startswith('.')]
    return files_with_extension


def filter(x, new_sampling_frequency, data_length_sec, lowcut, highcut):
    x1 = scipy.signal.resample(x, new_sampling_frequency * data_length_sec, axis=1)

    nyq = 0.5 * new_sampling_frequency
    b, a = sc.signal.butter(5, np.array([lowcut, highcut]) / nyq, btype='band')
    x_filt = sc.signal.lfilter(b, a, x1, axis=1)
    return np.float32(x_filt)


def process_file(read_dir, write_dir, lowcut, highcut,  raw_file_path):
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


def run_filter_preprocessor():
    with open('SETTINGS.json') as f:
        settings_dict = json.load(f)

    raw_data_path = settings_dict['path']['raw_data_path']
    processed_data_path = settings_dict['path']['processed_data_path'] + '/'+create_time_data_name(settings_dict)

    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)
    shutil.copy2('SETTINGS.json', processed_data_path + '/SETTINGS.json')

    highcut = settings_dict['preprocessor']['highcut']
    lowcut = settings_dict['preprocessor']['lowcut']

    #subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    # subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4']
    subjects=['Dog_5']
    for subject in subjects:
        print '>> filtering', subject
        read_dir = raw_data_path + '/' + subject
        write_dir = processed_data_path + '/' + subject

        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        raw_files = get_files_paths(read_dir)
        process_file(read_dir,write_dir,lowcut,highcut,raw_files[0])
        pool = Pool(6)
        part_f = partial(process_file, read_dir, write_dir, lowcut, highcut)
        pool.map(part_f, raw_files)


if __name__ == '__main__':
    run_filter_preprocessor()
