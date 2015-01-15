nfreq_bands = 7
win_length_sec = 120
stride_sec = 120
n_channels = 16
n_timesteps = (600-win_length_sec)/stride_sec + 1
global_pooling = 1

nkerns = [16, 32, 512]
recept_width = [1, 1]
stride = [1, 1]
pool_width = [1, 1]

n_params = 0

c1_input_width = n_timesteps
print 'c1:', nkerns[0], '@', ((n_timesteps - recept_width[0]) / stride[0] + 1) / pool_width[0]
n_params += (n_channels * nfreq_bands * recept_width[0] + 1) * nkerns[0]

c2_input_width = ((n_timesteps - recept_width[0]) / stride[0] + 1) / pool_width[0]
print 'c2:', nkerns[1], '@', ((c2_input_width - recept_width[1]) / stride[1] + 1) / pool_width[1]
n_params += (nkerns[0]*recept_width[1] + 1)*nkerns[1]

if global_pooling:
    f3_input_size = 6*nkerns[1]
else:
    f3_input_size = nkerns[1]*((c2_input_width - recept_width[1]) / stride[1] + 1) / pool_width[1]

n_params += f3_input_size * nkerns[2] + 1
print 'number of parameters', n_params

