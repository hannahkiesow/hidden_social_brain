import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

from project.helper_functions import set_random_seed
from preprocess import clean_volumes
from extract_variables import prepare_x_y, get_data_for_var, impute_random
from train_supervised_model import train_model, create_random_grid

#import handout
import joblib


# data preparation:




disc_volumes = pd.read_csv(sys.argv[1]) # e.g. './data/supervised/discovery_train_x.csv'
disc_y = pd.read_csv(sys.argv[2]) #e.g. './data/supervised/discovery_train_y.csv'
out_folder = sys.argv[3] #e.g. ./results/save/supervised/

db_path = sys.argv[4]
model_path = sys.argv[5]


_, _, rois = joblib.load('./data/dump_sMRI_socialbrain_sym_r2.5_s5')


# recoding:
# coding social as 1 nonsocial as 0 
recodings = {
    'Job SOC coding': lambda x: x,  # has to be done later
    'Friendships satisfaction': lambda x: 1 if x <= 2 else 0,  # as in ana_script
    'Family relationship satisfaction': lambda x: 1 if x == 2 else 0,  # as in ana
    # need to be adjusted later in different ways
    'Number in household': lambda x: x,
    'Number of older siblings': lambda x: 1 if x >= 1 else 0,  # as in hannah cross plot
    # as in hannah cross plot
    'Lifetime number of sexual partners': lambda x: 1 if x > 1 else 0,
    'Able to confide/Social support': lambda x: 1 if x == 5 else 0,  # as in ana
    'Leisure/social activities': lambda x: np.nan if x == 5 else 1 if x == 1 else 0,  # see ana
    'Loneliness, isolation': lambda x: 0 if x==1 else 1,  # lonely == nonsocial
    'Average total household income before tax': lambda x: 1 if x >= 4 else 0,
    'Private health care': lambda x: 1 if x <= 3 else 0,
    'BMI': lambda x: x,
    'Age at recruitment': lambda x: x,
    'Sex': lambda x: x,
    'Volume Normalized': lambda x: x,
    'Frequency of friend/family visits': lambda x: 0 if x <= 2 else 1,
}


assert (disc_volumes.eid == disc_y.eid).all()
# Leisure has to be recoded -7 to 7  # Leisure has to be recoded -7 to 7
disc_y.loc[disc_y['Leisure/social activities']
           == -7, 'Leisure/social activities'] = 7
X_train, y_train = prepare_x_y(disc_volumes, disc_y, recodings=recodings)


# X_train, X_test, y_train, y_test = train_test_split(X_prep, y_prep, test_size = 0.25)


# manual recoding:


def add_var_sex(y_train_, var):
    y_train_.loc[:, f'{var}+sex'] = y_train_[f'{var}'] + y_train_['Sex']*2


# social job + sex
top_10_jobs = y_train['Job SOC coding'].value_counts().iloc[:10]
social_job = [2314., 2315., 7111., 3211., 4123.]
y_train['social_job'] = y_train['Job SOC coding'].apply(
    lambda x: np.nan if x not in top_10_jobs else 1 if x in social_job else 0)

# y_test['social_job'] = y_test['Job SOC coding'].apply(
#    lambda x: np.nan if x not in top_10_jobs else 1 if x in social_job else 0)

y_train['social_job'] = impute_random(y_train['social_job'])


# social job + sex

add_var_sex(y_train, 'social_job')

# Friendship Satisfaction + sex
add_var_sex(y_train, 'Friendships satisfaction')

# Family Satisfaction + sex
add_var_sex(y_train, 'Family relationship satisfaction')

# Housholssize + sex

y_train['many_household'] = y_train.loc[:,
                                        'Number in household'].apply(lambda x: 1 if x > 2 else 0)

add_var_sex(y_train, 'many_household')

# living_alone + sex
y_train['living_alone'] = y_train.loc[:, 'Number in household'].apply(
    lambda x: 0 if x != 1 else 1)

add_var_sex(y_train, 'living_alone')

# Siblings + sex
add_var_sex(y_train, 'Number of older siblings')

# romantic partners +sex
add_var_sex(y_train, 'Lifetime number of sexual partners')

# social support + sex
add_var_sex(y_train, 'Able to confide/Social support')

# Leisure Social activities + sex
add_var_sex(y_train, 'Leisure/social activities')

# Loneliness + sex
add_var_sex(y_train, 'Loneliness, isolation')

# Income + sex
add_var_sex(y_train, 'Average total household income before tax')

# Health care + sex
add_var_sex(y_train, 'Private health care')


# data split + standadizing
set_random_seed()
X_train_original, y_train_original = clean_volumes(
    X_train, y_train, by_col=['Volume Normalized', 'BMI'])

z_transformer = StandardScaler().fit(X_train_original)
X_train_original = pd.DataFrame(z_transformer.transform(X_train_original))


def remove_outliers(df, y, cut_off):
    """
    Example
    -------

    >>> df_test = pd.DataFrame(dict(x=[0,-2.6,1], y=[-2.6,1,1]))
    >>> df_removed = remove_outliers(df_test, 2.5)
    >>> df_expected = df_test.iloc[-1:]
    >>> (df_removed == df_expected).all().all()
    True

    """
    good_inds = (df.abs() <= cut_off).all(axis=1)
    return df[good_inds], y[good_inds]


X_train_original, y_train_original = remove_outliers(
    X_train_original, y_train_original, 2.5)

logistic_grid = {
    'fit_intercept': [False],
    'C': np.logspace(-3, 3, 7),
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'multi_class': ['auto'],
    'solver': ['lbfgs'],
    'max_iter': [500]
}

forest_grid = create_random_grid()
n_bootstraps = 100
var_columns = [
    'social_job+sex',
    'Friendships satisfaction+sex',
    #'Family relationship satisfaction+sex',
    #'many_household+sex',
    'living_alone+sex',
    #'Number of older siblings+sex',
    'Lifetime number of sexual partners+sex',
    'Able to confide/Social support+sex',
    #'Leisure/social activities+sex',
    'Loneliness, isolation+sex',
    #'Average total household income before tax+sex',
    #'Private health care+sex',
]


X_train_original.reset_index(inplace=True, drop=True)
y_train_original.reset_index(inplace=True, drop=True)


for model, grid in zip([LogisticRegression(), RandomForestClassifier()],
                       [logistic_grid, forest_grid]):

    for var in var_columns:
        model_name = 'Logistic' if grid == logistic_grid else 'Forest'
        #doc = handout.Handout(f'/Users/student/Documents/Handout_Logistic_replication_2/handout_{var}_{model_name}')
        # doc.add_text(var)
        #doc.add_text(f'number of bootstraps: {n_bootstraps}, 5 outer (.8 train) fold, 5 inner')
        # doc.add_text('\n')
        for encoded in [True]:
            ages = [True] if encoded else [True]
            for age in ages:
                txt = var + '   '
                txt += 'latent' if encoded else '36 Rois'
                txt += '+age' if age else ''
                # doc.add_text(txt)
                # doc.add_text('logistic Regression') if grid == logistic_grid else doc.add_text('RandomForest')
                X_train, y_train = get_data_for_var(var,
                                                    X_train_original,
                                                    y_train_original,
                                                    encoded=encoded,
                                                    db_path=db_path,
                                                    model_path=model_path,
                                                    age=age,
                                                    )

                #doc.add_text(f'complete_training_n: {len(X_train)}')
                # doc.add_text('\n')
                ls_df_best_ests = train_model(model,
                                              X_train, y_train,
                                              hyper_param_grid=grid, n_bootstrap=n_bootstraps)
                #print_results(ls_df_best_ests, doc)

                # doc.add_text('\n')
                # doc.add_text('\n')
                # doc.show()
                AGE = 'age' if age == True else 'only_latent'
                df_out = pd.DataFrame(ls_df_best_ests)
                df_out.reset_index(inplace=True)
                df_out.to_json(
                    f'{out_folder}{"_".join(var.split("/"))}_{model_name}_{AGE}')
