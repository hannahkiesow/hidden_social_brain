import ast

from keras import Input, Model
from keras.engine.saving import load_model

from project.models.autoencoders import SymmetricAutoencoder


class TrainedAutoencoder():

    def __init__(self, autoencoder):
        self.autoencoder = autoencoder
        self.encoder = self.get_submodel_from('encode')
        self.decoder = self.get_submodel_from('decode')
        self.n_latent_variables = self.get_layer('encode')[-1].get_output_shape_at(0)[1]

    def get_submodel_from(self, name):
        submodel_layers = self.get_layer(name)
        inp_shape = (submodel_layers[0].get_input_shape_at(0)[1],)
        inp = Input(inp_shape, name='{}_input'.format(name))

        x = submodel_layers[0](inp)
        for layer in submodel_layers[1:]:
            x = layer(x)

        return Model(inp, x)

    def pred_with_chosen_variables(self, inp, chosen_variable='ALL'):
        encoded = self.encoder.predict(inp)

        if not chosen_variable == 'ALL':
            chose_latent_var = chosen_variable
            encoded[:, :chose_latent_var] = 0
            encoded[:, chose_latent_var + 1:] = 0

        pred = self.decoder.predict(encoded)
        return pred

    def pred_for_each_variable(self, inp):

        predictions = [self.pred_with_chosen_variables(inp=inp, chosen_variable=latent_var)
                       for latent_var in range(self.n_latent_variables)]

        return predictions

    def get_layer(self, name):
        layers = [layer for layer in self.autoencoder.layers if name in layer.name]
        return layers

    @classmethod
    def from_saved_model(cls, model_path):
        return cls(autoencoder=load_model(model_path))

    @classmethod
    def from_saved_weights(cls, model, weight_path):
        model.load_weights(weight_path, by_name=True)
        return cls(autoencoder=model)

    @classmethod
    def from_database(cls, df, model_path, model_id):
        """

        :param df: the DataFrame containing the hyperparams
        :param model_path: the complete path of the model to load in
        :param model_id: the id of the model inside of df
        :return: trained model of the id inside of df
        """
        def get_param_from_db(param):
            return ast.literal_eval(df.loc[model_id, [param]][0])

        model = SymmetricAutoencoder(input_shape=(36,), units=get_param_from_db('units'),
                                     use_biases=get_param_from_db('use_bias'),
                                     activations=get_param_from_db('activations'),
                                     batch_size=df.loc[model_id, ['batch_size']][0],
                                     reg_l1=df.loc[model_id, ['reg_l1']][0],
                                     reg_l2=df.loc[model_id, ['reg_l2']][0],
                                     reg_cov=df.loc[model_id, ['reg_cov']][0],
                                     tied_weights=df.loc[model_id, ['tied_weights']][0])

        model.encoding_layers = model.encoding_layers[:-1]

        from keras import Input, Model
        inp = Input((36,))

        for counter, layer in enumerate(model.encoding_layers + model.decoding_layers):
            if counter == 0:
                x = layer(inp)
            else:
                x = layer(x)

        model = Model(inp, x)
        return cls.from_saved_weights(model, model_path)
