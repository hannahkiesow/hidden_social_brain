import sys
from itertools import product

from keras import metrics
from keras.optimizers import RMSprop, SGD, Adam, Adagrad

from project.models.autoencoders import SymmetricAutoencoder
from project.helper_functions import set_random_seed, create_hyperparam_list, z_standardize

import joblib

data_path = sys.argv[1]  # e.g './data/discovery_data/train_dump_sMRI_socialbrain_sym_r2.5_s5'
save_path = sys.argv[2]  # e.g. './results/save/unsupervised/'

set_random_seed()

vols_raw = joblib.load(data_path)
vols_standard = z_standardize(vols_raw)

default_init_params = {
    'units': [[15, 36]],
    'activations': [[None, None]],
    'use_biases': [[False]],
    'reg_l1': [0],
    'reg_l2': [0],
    'reg_cov': [0],
    'tied_weights': [False],
    'input_shape': [(36,)],
    'batch_size': [36],
}

default_train_params = {
    'name': 'default_name',
    'x': [vols_standard],
    'epochs': [15],
    'perc_validation': [.1],
    'loss': ['mean_squared_error'],
    'optimizer_func': [RMSprop],
    'learning_rate': [0.001],
    'list_metrics': [[metrics.mae]],
    'save_dir': [save_path]
}

experiments = [
    ({'use_biases': [[True, True], [False, False]]},
     {'epochs': [15, 30], 'learning_rate': [0.1, 0.01, 0.001, 0.0001],
      'optimizer_func': [RMSprop, SGD, Adam, Adagrad], 'name': ['hyperparameter']}),
    ({}, {'name': ['linear']}),
    ({'reg_l1': [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]}, {'name': ['l1']}),
    ({'reg_l2': [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]}, {'name': ['l2']}),
    ({'reg_cov': [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]},
     {'name': ['covariance']}),
    ({'tied_weights': [True], 'activations': list(
        product(['relu', 'tanh', None], repeat=2))}, {'name': ['tied']}),
    ({'tied_weights': [True, False],
      'units': [[25, 15, 25, 36]],
      'activations': list(product(['relu', 'tanh', None], repeat=4))}, {'name': ['4-Layer']}),
    ({'tied_weights': [True, False],
      'units': [[25, 20, 15, 20, 25, 36]],
      'activations': [[None, None, None, None, None, None],
                      ['relu', 'relu', 'relu', 'relu', 'relu', None],
                      ['relu', 'relu', 'relu', 'relu', None, None],
                      ['relu', 'relu', 'relu', 'tanh', None, None]
                      ]}, {'name': ['6-Layer']})
]

init_params, fit_params = [], []
for change_init_params, change_train_params in experiments:
    init_param, fit_param = create_hyperparam_list(default_init_params, default_train_params,
                                                   change_init_params, change_train_params)

    init_params += init_param
    fit_params += fit_param
for init_param, fit_param in zip(init_params, fit_params):
    set_random_seed()
    auto = SymmetricAutoencoder(**init_param)
    auto.fit(**fit_param)
