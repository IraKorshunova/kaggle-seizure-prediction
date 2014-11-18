import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def scale_across_features(x, x_test=None, scalers=None):
    n_channels = x.shape[1]
    n_fbins = x.shape[2]
    n_timesteps = x.shape[3]
    flatten_dim = n_channels * n_fbins * n_timesteps
    x = x.reshape(x.shape[0], flatten_dim)

    if x_test is not None:
        x_complete = np.vstack((x, x_test.reshape(x_test.shape[0], flatten_dim)))
    else:
        x_complete = x

    if scalers is None:
        scalers = StandardScaler()
        scalers.fit(x_complete)

    x = scalers.transform(x)
    x = x.reshape(x.shape[0], n_channels, n_fbins, n_timesteps)
    return x, scalers


def scale_across_time(x, x_test=None, scalers=None):
    n_examples = x.shape[0]
    n_channels = x.shape[1]
    n_fbins = x.shape[2]
    n_timesteps = x.shape[3]
    if scalers is None:
        scalers = [None] * n_channels

    print x.shape
    for i in range(n_channels):
        xi = np.transpose(x[:, i, :, :], axes=(0, 2, 1))
        xi = xi.reshape((n_examples * n_timesteps, n_fbins))

        if x_test is not None:
            xi_test = np.transpose(x_test[:, i, :, :], axes=(0, 2, 1))
            xi_test = xi_test.reshape((x_test.shape[0] * n_timesteps, n_fbins))
            xi_complete = np.vstack((xi, xi_test))
        else:
            xi_complete = xi

        if scalers[i] is None:
            scalers[i] = StandardScaler()
            scalers[i].fit(xi_complete)

        xi = scalers[i].transform(xi)

        xi = xi.reshape((n_examples, n_timesteps, n_fbins))
        xi = np.transpose(xi, axes=(0, 2, 1))
        x[:, i, :, :] = xi
    return x, scalers
