from keras.engine import InputSpec, Layer
from keras.layers import Dense
import keras.backend as K


class DenseTied(Dense):
    def __init__(self, tied_layer, **kwargs):
        self.tied_layer = tied_layer
        super().__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.tied_layer.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        self.W = K.transpose(self.tied_layer.kernel)
        output = K.dot(inputs, self.W)
        if self.use_bias:
            print('out', output)
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output


class CovarianceLoss(Layer):

    def __init__(self, batch_size, weight, **kwargs):
        self.weight = weight
        self.batch_size = batch_size
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x = inputs - K.mean(inputs, axis=0)  # minus mean of each var over samples
        cov = K.dot(K.transpose(x), x)
        cov -= cov * K.eye(K.shape(cov)[0])  # minus diagonal
        cov = K.square(cov)  # to get positive values only
        cov *= (1 / self.batch_size)
        loss = K.sum(cov) * self.weight  # to change influence of covariance on loss

        self.add_loss(loss, inputs=inputs)

        return inputs

    def get_output_shape_for(self, input_shape):
        return input_shape
