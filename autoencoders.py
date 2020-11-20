from keras import Input, Model
from keras.layers import Dense
from keras.regularizers import l1_l2

from .keras_extensions.layers import DenseTied, CovarianceLoss
from ..helper_functions import number_to_equal_digit, save_model


class SymmetricAutoencoder:

    def __init__(self, input_shape, units, use_biases, activations,
                 batch_size, reg_l1, reg_l2, reg_cov, tied_weights):

        """
        :param input_shape: The input shape, without the batch_size
        :param units: a list of the unit sizes for each layer
        :param use_biases: a list of whether to use a bias for each layer. If only one value, then applied to all.
        :param activations: a list of the activations for each layer or one value, then applied to all
        :param batch_size: batch size used, needs to be defined here for covariance. Will be used for train as well.
        :param reg_l1: lambda for regularizing l1, if 0 then no regularization.
        :param reg_l2: lambda for regularizing l2, if 0 then no regularization.
        :param reg_cov: the lambda for covariance, if 0 then no covariance
        :param tied_weights: Boolean whether the encoder and decoder have tied weights
        :return: Keras model of the specified autoencoder
        """

        self.input_shape = input_shape
        self.length = len(units)
        if not self.length % 2 == 0:
            raise ValueError(f'Symmetric Autoencoder has to have an even lenght of units not {self.length}')

        self.units = units
        self.use_biases = self.check_inp_lst(use_biases)
        self.activations = self.check_inp_lst(activations)
        self.batch_size = batch_size
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.reg_cov = reg_cov
        self.tied_weights = tied_weights
        self.encoding_layers = []
        self.decoding_layers = []
        self.autoencoder = None

        self.initialize_layers()  # fills encoding and decoding layers

    def initialize_layers(self):
        """
        initializes all layers => self.encoder and self.decoder are fille with there respective layers


        :return:
        """
        for counter, (unit_size, use_bias, activation) in enumerate(
                zip(self.units, self.use_biases, self.activations)):

            number = number_to_equal_digit(counter, 3)

            if counter < self.length / 2 - 1:  # encoding
                name = f'encoder_{number}'
                layer = Dense(units=unit_size, use_bias=use_bias, activation=activation, name=name)
                self.encoding_layers.append(layer)

            elif counter == self.length / 2 - 1:  # latent embedding
                name = f'encoder_{number}_z'
                layer = Dense(units=unit_size, use_bias=use_bias, activation=activation,
                              kernel_regularizer=l1_l2(self.reg_l1, self.reg_l2), name=name)
                self.encoding_layers.append(layer)
                self.encoding_layers.append(CovarianceLoss(batch_size=self.batch_size, weight=self.reg_cov))

            else:  # decoder
                name = f'decoder_{number}'
                if not self.tied_weights:
                    layer = Dense(units=unit_size, use_bias=use_bias, activation=activation, name=name)
                    self.decoding_layers.append(layer)
                else:
                    # starting from -2 (because -1 is the Covariance Layer)
                    tied_layer = self.encoding_layers[-int(counter - self.length / 2) - 2]
                    layer = DenseTied(units=unit_size, tied_layer=tied_layer,
                                      use_bias=use_bias, activation=activation, name=name)
                    self.decoding_layers.append(layer)

    def fit(self, x, perc_validation, epochs, optimizer_func,
            learning_rate, loss, list_metrics, save_dir='default', name='default_name'):
        self.compile(optimizer_func, learning_rate, loss, list_metrics)

        history = self.autoencoder.fit(x=x, y=x, epochs=epochs, batch_size=self.batch_size,
                                       shuffle=True, validation_split=perc_validation)
        if save_dir != 'default':
            save_model(model=self.autoencoder, save_dir=save_dir, name=name, history=history,
                       units=self.units, use_bias=self.use_biases, activations=self.activations,
                       reg_l1=self.reg_l1, reg_l2=self.reg_l2, reg_cov=self.reg_cov, tied_weights=self.tied_weights,
                       optimizer_func=optimizer_func, learning_rate=learning_rate, epochs=epochs,
                       batch_size=self.batch_size)

        return self.autoencoder

    def compile(self, compiler_func, learning_rate, loss, list_metrics):

        inp = Input(self.input_shape, name='input')
        x = self.encoding_layers[0](inp)
        for layer in self.encoding_layers[1:] + self.decoding_layers:
            x = layer(x)

        self.autoencoder = Model(inp, x)
        self.autoencoder.compile(optimizer=compiler_func(lr=learning_rate), loss=loss, metrics=list_metrics)

    def check_inp_lst(self, inp_lst):
        if len(inp_lst) == self.length:
            return inp_lst
        elif len(inp_lst) == 1:
            return inp_lst * self.length
        else:
            raise ValueError(f'{inp_lst} must have a len() of 1 or {self.length}')
