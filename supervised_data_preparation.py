import joblib
import pandas as pd
import numpy as np
from project.helper_functions import set_random_seed

set_random_seed()

#train_data = joblib.load('./data/replication_data/test_dump_sMRI_socialbrain_sym_r2.5_s5')
#train_T1 = joblib.load('./data/replication_data/test_T1_subnames_dump_sMRI_socialbrain_sym_r2.5_s5')

#train_data = pd.read_csv('./data/discovery_data/train.csv', index_col=False)
#train_data.columns = ['eid'] +  list(train_data.iloc[:,1:].columns.values)
#train_T1 = train_data.eid
#train_T1 = [i[2:-1] for i in train_T1]
for data_set in ['discovery', 'replication']:
    for train_or_test in ['train', 'test']:
        print(train_or_test)

        train_data = joblib.load(f'./data/{data_set}_data/{train_or_test}_dump_sMRI_socialbrain_sym_r2.5_s5')
        
        train_T1 = joblib.load(f'./data/{data_set}_data/{train_or_test}_T1_subnames_dump_sMRI_socialbrain_sym_r2.5_s5')


        ukbb = pd.read_csv('./data/supervised/ukb_add1_holmes_merge_brain.csv')
        df_text = pd.read_csv('./data/supervised/ukbbids_social_brain.txt', sep='\t', header=0)
        x_saving_path = f'./data/supervised/{data_set}_{train_or_test}_x.csv'
        y_saving_path = f'./data/supervised/{data_set}_{train_or_test}_y.csv'



        df_text.columns = ['series_number', 'raw_feature']
        to_int = lambda x: np.array(list(map(int, x)))
        df_training = pd.DataFrame(train_data)
        #df_training = train_data
        df_training['eid'] = to_int(train_T1)

        # From Danilo
        T1_subnames_int = to_int(train_T1)
        inds = np.searchsorted(T1_subnames_int, ukbb.eid)
        inds_mri = []
        source_array = T1_subnames_int
        for _, sub in enumerate(ukbb.eid):
            i_found = np.where(sub == source_array)[0]
            if len(i_found) == 0:
                continue
            inds_mri.append(i_found[0])  # take first found subject
        b_inds_ukbb = np.in1d(ukbb.eid, source_array[inds_mri])


        print('%i matched matrices between data and UKBB found!' % np.sum(
            source_array[inds_mri] == ukbb.eid[b_inds_ukbb]))


        df_uk = ukbb.loc[b_inds_ukbb]
        df = df_training.loc[inds_mri]
        assert all(df.eid.values == df_uk.eid.values)


        def extract_classes(row):
            feature_labels = row['raw_feature']
            feature, *labels = feature_labels.split('(')
            labels = [label[0].upper() for label in labels]
            feature = feature[:-1]  # get rid of space at the end
            return feature, labels


        df_text['feature'], df_text['labels'] = zip(*df_text.apply(extract_classes, axis=1))

        number_to_feature = {}


        def append_my_dict(row):
            number_to_feature[row.series_number] = row.feature


        df_text.apply(append_my_dict,
                    axis=1)

        df_processed = df_uk
        df_processed.columns = [number_to_feature[col] if col in number_to_feature else col for col in df_processed.columns]

        columns = ['eid'] + list(number_to_feature.values())
         
        df_processed = df_processed[columns]  # exclude all variables not as features


        y = df_processed.copy()
        y.index = y.eid
        y = y.loc[:, y.columns[1:]]
        x = df.copy()
        x.index = x.eid
        x = x.loc[:, x.columns[:-1]]

        x.to_csv(x_saving_path)
        y.to_csv(y_saving_path)
