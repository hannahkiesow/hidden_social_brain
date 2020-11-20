import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from project.helper_functions import set_random_seed

set_random_seed()
complete_data_path = './data/dump_sMRI_socialbrain_sym_r2.5_s5'
save_data_folder = '/'.join(complete_data_path.split('/')[:-1])
T1_subnames, DMN_vols, rois = joblib.load(complete_data_path)

discovery_data, replication_data, T1_discovery, T1_replication = train_test_split(DMN_vols, T1_subnames, test_size=0.5)

for data_set, name in zip([(discovery_data, T1_discovery), (replication_data, T1_replication)],
                          ['discovery_data', 'replication_data']):

    data, T1 = data_set
    train_data, test_data, train_T1, test_T1 = train_test_split(data, T1, test_size=0.2)
    joblib.dump(train_data, f'{save_data_folder}/{name}/train_dump_sMRI_socialbrain_sym_r2.5_s5')
    joblib.dump(test_data, f'{save_data_folder}/{name}/test_dump_sMRI_socialbrain_sym_r2.5_s5')
    joblib.dump(train_T1, f'{save_data_folder}/{name}/train_T1_subnames_dump_sMRI_socialbrain_sym_r2.5_s5')
    joblib.dump(test_T1, f'{save_data_folder}/{name}/test_T1_subnames_dump_sMRI_socialbrain_sym_r2.5_s5')
    train = pd.DataFrame(train_data, index = train_T1)
    test = pd.DataFrame(test_data, index = test_T1)
    train.to_csv(f'{save_data_folder}/{name}/train.csv')
    
    test.to_csv(f'{save_data_folder}/{name}/test.csv')
