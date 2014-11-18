import json
from utils.config_name_creator import create_cnn_model_name
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import PIL.Image as Image


def paint_filter(patient_name, model_path):
    with open(model_path + '/' + patient_name + '.pickle', 'rb') as f:
        state = cPickle.load(f)

    W1 = state['weights'][0]
    print W1.shape
    nkerns = W1.shape[0]
    filters = [W1]
    fig = plt.figure()
    for w in filters:
        for fm in range(w.shape[0]):
            ax=fig.add_subplot(1, nkerns, fm)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.imshow(w[fm, 0, :, :], interpolation='none')
    plt.show()


if __name__ == '__main__':
    with open('SETTINGS.json') as f:
        settings_dict = json.load(f)

    s = '0.81475_GPU_lowcu0.1nfreq6highc180win_l60featusumloggloba0recep[1, 2]activreludropo[0.2, 0.5]overl0strid[1, 1]train10weigh0.01nkern[16, 32, 128]pool_[1, 1]l2_re0.0001valid10max_i150000rando1'
    model_path = settings_dict['path']['model_path'] + '/' + create_cnn_model_name(settings_dict)
    names = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    for patient_name in names:
        paint_filter(patient_name, model_path)
