####################### ReadMe ########################
#
# This is similar to volume data preprocessing. Here I also concat the 'sample' field I generated for division of training to ...
#....further internal validation and ensemble set.
####################### ------ ########################


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random

## Define paths

# path = '/home/...'  # Give parent path

data_dir = os.path.join(os.sep, path)  

train1_filename = 'train_count_grey.csv'
train2_filename = 'train_intensity_grey.csv'
val1_filename = 'val_count_grey.csv'
val2_filename = 'val_intensity_grey.csv'
test1_filename ='test_count_grey.csv'
test2_filename ='test_intensity_grey.csv'
all_data_loc = 'all_volumetrix_processed.csv'


train1= pd.read_csv(data_dir+train1_filename, index_col=False)
train2= pd.read_csv(data_dir+train2_filename, index_col=False)
val1= pd.read_csv(data_dir+val1_filename, index_col=False)
val2= pd.read_csv(data_dir+val2_filename, index_col=False)
test1= pd.read_csv(data_dir+test1_filename, index_col=False)
test2= pd.read_csv(data_dir+test2_filename, index_col=False)


all_data =  pd.read_csv(data_dir+all_data_loc, index_col=False)
all1 = train1.append(val1.append(test1))
all2 = train2.append(val2.append(test2))
all1 = all1.fillna(0).copy()
all2 = all2.fillna(0).copy()

#removing constant columns
all1 = all1.loc[:,all1.apply(pd.Series.nunique) != 1]
all1 = all1.drop(['Unnamed: 0'], axis=1).copy()
all1.columns = ['cnt_reg_' + str(col) for col in all1.columns]

all2 = all2.loc[:,all2.apply(pd.Series.nunique) != 1]
all2 = all2.drop(['Unnamed: 0'], axis=1).copy()
all2.columns = ['int_reg_' + str(col) for col in all2.columns]


#adding gender,target and sample data
all1_full = pd.merge(all_data[['SUBJECTKEY','INTERVIEW_AGE','sex_bin','STUDY_COHORT_NAME','residual_fluid_intelligence_score','sample']],                              all1, left_on='SUBJECTKEY', right_on='cnt_reg_ID')
all1_full = all1_full.drop(['cnt_reg_ID'], axis=1).copy()

all2_full = pd.merge(all_data[['SUBJECTKEY','INTERVIEW_AGE','sex_bin','STUDY_COHORT_NAME','residual_fluid_intelligence_score','sample']],                              all2, left_on='SUBJECTKEY', right_on='int_reg_ID')
all2_full = all2_full.drop(['int_reg_ID'], axis=1).copy()


# ----
# ### Normalizing and checking for correlation

cols_asIDV = all1_full.columns.difference(['SUBJECTKEY','STUDY_COHORT_NAME', 'sample','residual_fluid_intelligence_score'])

idv_scaler = MinMaxScaler(feature_range=(0, 1))
idv_scaler.fit(all1_full[cols_asIDV])
all1_full[cols_asIDV] = idv_scaler.transform(all1_full[cols_asIDV])

cols_asIDV2 = all2_full.columns.difference(['SUBJECTKEY','STUDY_COHORT_NAME','sample','residual_fluid_intelligence_score'])

idv_scaler = MinMaxScaler(feature_range=(0, 1))
idv_scaler.fit(all2_full[cols_asIDV2])
all2_full[cols_asIDV2] = idv_scaler.transform(all2_full[cols_asIDV2])



## Saving both to disk
all1_full.to_csv(data_dir+'all_countGrey_data.csv', sep=',',index=False)
all2_full.to_csv(data_dir+'all_instensityGrey_data.csv', sep=',',index=False)

