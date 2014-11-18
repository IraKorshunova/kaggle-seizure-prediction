from layers.hidden_layer import HiddenLayer
from layers.conv_layer import ConvPoolLayer
from glob_pool_layer import GlobalPoolLayer


class FeatureExtractor(object):
    def __init__(self, rng, input, nkerns, recept_width, pool_width, stride, training_mode, dropout_prob, activation,
                 weights_variance, n_channels, n_timesteps, n_fbins, global_pooling):
        self.layer0 = ConvPoolLayer(rng, input=input,
                                    image_shape=(None, 1, n_channels * n_fbins, n_timesteps),
                                    filter_shape=(nkerns[0], 1, n_fbins * n_channels, recept_width[0]),
                                    poolsize=(1, pool_width[0]), activation=activation[0],
                                    weights_variance=weights_variance, subsample=(1, stride[0]))

        input_layer1_width = ((n_timesteps - recept_width[0]) / stride[0] + 1) / pool_width[0]
        self.layer1 = ConvPoolLayer(rng, input=self.layer0.output,
                                    image_shape=(None, nkerns[0], 1, input_layer1_width),
                                    filter_shape=(nkerns[1], nkerns[0], 1, recept_width[1]),
                                    poolsize=(1, pool_width[1]), activation=activation[1],
                                    weights_variance=weights_variance, subsample=(1, stride[1]))

        if global_pooling:
            self.glob_pool = GlobalPoolLayer(self.layer1.output)
            layer2_input = self.glob_pool.output.flatten(2)

            input_layer2_shape = nkerns[1] * 6
            self.layer2 = HiddenLayer(rng=rng, input=layer2_input,
                                      n_in=input_layer2_shape, n_out=nkerns[2],
                                      training_mode=training_mode,
                                      dropout_prob=dropout_prob, activation=activation[2],
                                      weights_variance=weights_variance)
        else:
            layer2_input = self.layer1.output.flatten(2)
            input_layer2_size = ((input_layer1_width - recept_width[1]) / stride[1] + 1) / pool_width[1]
            self.layer2 = HiddenLayer(rng=rng, input=layer2_input,
                                      n_in=nkerns[1] * input_layer2_size, n_out=nkerns[2],
                                      training_mode=training_mode,
                                      dropout_prob=dropout_prob, activation=activation[2],
                                      weights_variance=weights_variance)

        self.output = self.layer2.output
        self.weights = self.layer0.weights + self.layer1.weights + self.layer2.weights