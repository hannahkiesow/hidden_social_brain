import numpy as np
import pandas as pd
import tensorflow as tf
import random as rn
from keras import backend as K
import warnings

from sklearn.model_selection import ParameterGrid


def set_random_seed():
    """
    for more information see
    https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    """
    warnings.warn("For reproducible results use CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0")
    # all random seeds were chosen by the following function:
    # # random state chose by randint(0, 1_000_000_000)
    np.random.seed(638717371)
    rn.seed(396537264)
    # turning off parallel computation, because it leads to non-deterministic results
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1)

    tf.set_random_seed(168393118)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


def number_to_equal_digit(number, number_of_digits):
    """

    :param number: The number that should be converted
    :param number_of_digits: How many digits should the number get 2 for 01, 3 for 001, etc.
    :return: str of original number with equal digits to number_of_digits. Example (1,3) -> 001
    """

    if len(str(number)) < number_of_digits:
        return number_to_equal_digit(f'0{number}', number_of_digits)

    elif len(str(number)) > number_of_digits:
        raise ValueError(f'The length of str(number) should not exceed numbers_of_digits. \n'
                         f' Here number is {number} and numbers_of_digits is {number_of_digits}')

    else:
        return number


def z_standardize(volume_data):
    df = pd.DataFrame(volume_data)
    cols = list(df.columns)
    df_out = pd.DataFrame()
    for col in cols:
        df_out[col] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

    return df_out.values


def save_model(model, save_dir, name, history, units, use_bias, activations,
               reg_l1, reg_l2, reg_cov, tied_weights, optimizer_func, learning_rate, epochs, batch_size):
    df_add = pd.DataFrame(
        {'history': [history.history],
         'units': [units],
         'use_bias': [use_bias],
         'activations': [activations],
         'reg_l1': [reg_l1],
         'reg_l2': [reg_l2],
         'reg_cov': [reg_cov],
         'tied_weights': [tied_weights],
         'optimizer_func': [optimizer_func],
         'learning_rate': [learning_rate],
         'epochs': [epochs],
         'batch_size': [batch_size]
         })

    try:
        df_db = pd.read_csv(f'{save_dir}{name}.txt', index_col=False)
        df_db = df_db.append(df_add, ignore_index=True, sort=True)
        current_id = len(df_db) - 1

    except FileNotFoundError:
        current_id = 0
        df_db = df_add

    df_db.to_csv(f'{save_dir}{name}.txt', index=False)
    auto_dir = save_dir + '/model/{}autoencoder_'.format(name) + str(current_id) + '.h5'
    auto_dir_weights = save_dir + '/model/{}autoencoder_weights_'.format(name) + str(current_id) + '.h5'
    model.save(auto_dir)
    model.save_weights(auto_dir_weights)


def create_hyperparam_list(default_init_params, default_train_params,
                           change_init_params, change_train_params):
    all_params = {}
    for key, value in default_init_params.items():
        all_params[key] = change_init_params[key] if key in change_init_params else value

    for key, value in default_train_params.items():
        all_params[key] = change_train_params[key] if key in change_train_params else value

    all_params_grid = ParameterGrid(all_params)

    init_params = []
    train_params = []
    for parm_dict in all_params_grid:
        init_param = {key: value for key, value in parm_dict.items() if key in default_init_params}
        train_param = {key: value for key, value in parm_dict.items() if key in default_train_params}

        init_params.append(init_param)
        train_params.append(train_param)

    return init_params, train_params
