import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from project.helper_functions import set_random_seed

from sklearn.model_selection import KFold, ShuffleSplit, GridSearchCV, cross_val_score, cross_validate, train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import resample
from sklearn.metrics import accuracy_score


def train_model(model, X_train, y_train, hyper_param_grid, n_bootstrap=100):

    ls_df_best_ests = []
    set_random_seed()
    outer_fold = ShuffleSplit(n_splits=5, train_size=.8)
    for n_fold, (train_outer_index, test_outer_index) in enumerate(outer_fold.split(X_train, y_train)):

        X_outer_test = pd.DataFrame(X_train).iloc[test_outer_index]
        y_outer_test = y_train.iloc[test_outer_index]

        for _ in range(n_bootstrap):
            train_bootstrap_indx = resample(train_outer_index, replace=True)
            X_bootstrap_train = pd.DataFrame(
                X_train).iloc[train_bootstrap_indx]
            y_bootstrap_train = y_train.iloc[train_bootstrap_indx]

            LR = GridSearchCV(model,
                              param_grid=hyper_param_grid, cv=7, n_jobs=3, return_train_score=True)

            LR.fit(X_bootstrap_train, y_bootstrap_train)
            best_ind = LR.best_index_
            df = pd.DataFrame(LR.cv_results_)

            # getting rid of not needed columns to save memory
            columns = [col
                       for col in df.columns
                       if (('test' in col or 'train' in col)
                           and not ('std' in col
                                    or 'mean' in col
                                    or 'rank' in col))]

            df = df.loc[best_ind, columns]
            df['best_est'] = LR.best_estimator_
            df['outer_test_score'] = LR.best_estimator_.score(
                X_outer_test, y_outer_test)
            df['inner_score'] = LR.best_estimator_.score(
                X_bootstrap_train, y_bootstrap_train)

            df['n_fold'] = n_fold
            class_rep = classification_report(
                y_outer_test, LR.best_estimator_.predict(X_outer_test), output_dict=True)
            for i in range(len(np.unique(y_outer_test))):
                df[f'f1_class_{i}'] = class_rep[str(float(i))]['f1-score']

            ls_df_best_ests.append(df)

    return ls_df_best_ests


def create_random_grid():
    """
    creates grid used for randomgridCV
    :return: grid with hyperparameters
    """
    # number of trees
    n_estimators = [100]  # np.arange(start=100, stop=200, step=50)

    # number of features
    max_features = ['auto']  # ['auto', 'sqrt']

    # number of levels
    max_depth = [2, 6]  # np.arange(start=2, stop=10, step=1)
    # max_depth.append(None)

    # min number of samples required to split a node
    # min_samples_split = [2, 5, 10]
    min_samples_split = [2, 6]  # [2, 5]

    # min number of samples at each leaf node
    min_samples_leaf = [2, 6]

    # method of selecting samples for training each tree
    bootstrap = [True]

    # create random grid.., dn)Â¶
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    return random_grid
