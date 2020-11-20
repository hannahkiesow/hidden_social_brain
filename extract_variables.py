import pandas as pd
import numpy as np
from functools import partial
from project.helper_functions import set_random_seed
from project.models.autoencoders import SymmetricAutoencoder
from project.models.trained_autoencoder import TrainedAutoencoder

from sklearn.preprocessing import StandardScaler
from keras.optimizers import RMSprop
from keras import metrics

from preprocess import impute_random, balance_x_y


def recode_variables(row, recodings):
    recoded_dict = {name: func(row[name]) for name, func in recodings.items()}
    return pd.Series(recoded_dict)



def get_from_file_autoencoder(db_path, model_path):

    #db_path = './results/save/unsupervised/linear.txt'
    #model_path = './results/save/unsupervised/model/linearautoencoder_0.h5' 
    df = pd.read_csv(db_path)
    trained_autoencoder = TrainedAutoencoder.from_database(df, model_path, model_id=0)
    return trained_autoencoder





def prepare_x_y(volumes, y, recodings):
    set_random_seed()
    volumes_ = volumes.iloc[:,1:] # 0th column is eid, not needed here

    y_selected = (y
                    .iloc[:, 1:] # Oth column is eid, not needed here 
                    .applymap(lambda x: np.nan if x < 0 else x) # below zero codes for missing reasons
                    .apply(impute_random)
                    .apply(partial(recode_variables,recodings=recodings), axis=1)
                    
                 )
    return volumes_, y_selected



def get_data_for_var(var,X_train_original, y_train_original, encoded, db_path, model_path, age=False):
    set_random_seed()
    
    train_nan = y_train_original[var].isnull()

    X_train_bal, y_train_bal, y_train_ind = balance_x_y(X_train_original.loc[train_nan==False, :], y_train_original.loc[train_nan==False, var])

    if encoded:
        trained_autoencoder = get_from_file_autoencoder(db_path, model_path)
        X_train_bal = trained_autoencoder.encoder.predict(X_train_bal)
        if age: 
            X_train_bal = pd.DataFrame(X_train_bal)
            X_train_bal[15] = StandardScaler().fit_transform(y_train_original.loc[y_train_ind, ['Age at recruitment']])
            #X_train_bal[16] = y_train_original.loc[y_train_ind, ['Sites']].reset_index(drop=True)
            
            X_train_bal = X_train_bal.values
    return X_train_bal, y_train_bal
