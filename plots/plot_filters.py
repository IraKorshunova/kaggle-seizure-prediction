import json
from utils.config_name_creator import create_cnn_model_name
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import PIL.Image as Image


def paint_filter(patient_name, model_path):
    with open(model_path + '/' + patient_name + '.pickle', 'rb') as f:
        state = cPickle.load(f)

    # first layer
    W1 = state['weights'][0]
    width, heights = W1.shape[3], W1.shape[2]
    n_filters = W1.shape[0]
    n_fbins = state['params']['n_fbins']
    print W1.shape
    x = np.zeros((heights, width * n_filters))
    for i in range(n_filters):
        x[:, i] = np.reshape(W1[i, 0, :, :], heights * width)

    ax = plt.gca()
    ax.set_yticks(range(0, heights, n_fbins))
    ax.yaxis.grid(True, which='major', linestyle='-')
    plt.imshow(x, interpolation='none')
    plt.show()


if __name__ == '__main__':
    with open('SETTINGS.json') as f:
        settings_dict = json.load(f)

    s1 = '0.81448_nfreq6featumeanloghighc180lowcu0.1win_l60strid60globa0recep[1, 2]use_t0activ[urelu, urelu, urelu]dropo[0.2, 0.5]overl0strid[1, 1]train10weigh0.01scale0nkern[16, 32, 128]pool_[1, 1]l2_re0.0001valid10max_i150000rando1'
    s2 = '0.80192_nfreq8featumeanlog_stdhighc180lowcu0.1win_l120strid120globa1recep[1, 1]use_t0activ[urelu, urelu, utanh]dropo[0.3, 0.6]overl4strid[1, 1]train10weigh0.01scale1nkern[16, 32, 512]pool_[1, 1]l2_re0.0001valid10max_i150000rando1'
    model_path = settings_dict['path']['model_path'] + '/' + s2  # create_cnn_model_name(settings_dict)
    names = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    for patient_name in names:
        paint_filter(patient_name, model_path)
