import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from nilearn import signal

from project.helper_functions import set_random_seed

def impute_random(arr_in):
    set_random_seed()
    arr = arr_in.copy() # to not change the input
    arr_is_nan = np.isnan(arr) # 1 if nan else 0
    n_nan = sum(arr_is_nan)
    arr_without_nan = arr[arr_is_nan==False]
    arr_fill = np.random.choice(arr_without_nan, size=n_nan)
    arr[arr_is_nan] = arr_fill
    return arr

def balanced_sample(column):
    out = pd.Series(np.zeros(len(column)))
    value_counts = column.value_counts()
    n_smalles_value = min(value_counts)
    for value, n_value in value_counts.to_dict().items():
        select_from_column = np.zeros(n_value)
        select_from_column[:n_smalles_value] = 1
        np.random.shuffle(select_from_column)
        out[column == value] = select_from_column

    return out

def balance_x_y(x, y):
    set_random_seed()
    y_ind = y.index.values
    x = x.reset_index(drop=True) #temp
    y = y.reset_index(drop=True) #temp
    select = balanced_sample(y)
    y = y[select == 1]
    x = x.loc[select == 1]
    y_ind = y_ind[select == 1]
    return x, y, y_ind


def clean_volumes(volumes,y, by_col):
    cleaner = lambda col_name: np.atleast_2d(StandardScaler().fit_transform(np.nan_to_num(y[col_name].values[:, None])))
    conf_mat = np.hstack([cleaner(col) for col in by_col])
    
    volumes_deconf = signal.clean(volumes.values, confounds=conf_mat, detrend=False, standardize=False)
    return pd.DataFrame(volumes_deconf, index = volumes.index), y

