####################### ReadMe ########################
#
# 1) Parent path contains 5 folders > train,val,test,models,scores. val stands for validation
# 2) Respective volume scores and *nii.gz files of segments train, validation,test are in respective folders.
# 3) Volume scores of each ROI provided by challengers are saved as volumetrix_data_[segment]
# 4) Fluid intelligence score are saved for each as [segment]_y where test_y is self created and contains no value for this column
#
####################### ------ ########################

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random



## Define paths

# path = '/home/...'  # Give parent path

data_dir = os.path.join(os.sep, path)  


trainy_filename = 'train/train_y.csv'
valy_filename = 'val/val_y.csv'
testy_filename ='test/test_y.csv'

trainx_filename = 'train/volumetrix_data_train.csv'
valx_filename = 'val/volumetrix_data_val.csv'
testx_filename = 'test/volumetrix_data_test.csv'

## Loading datasets

trainy= pd.read_csv(data_dir+trainy_filename, index_col=False)
valy= pd.read_csv(data_dir+valy_filename, index_col=False)
testy= pd.read_csv(data_dir+testy_filename, index_col=False)
trainx =  pd.read_csv(data_dir+trainx_filename, index_col=False)
valx =  pd.read_csv(data_dir+valx_filename, index_col=False)
testx =  pd.read_csv(data_dir+testx_filename, index_col=False)


## Append all

all_data = trainx.append(valx).append(testx).copy()
all_data['sex_bin'] = np.where(all_data['GENDER']=='F', 1, 0)
all_data = all_data.drop(['BTSV01_ID','DATASET_ID','GENDER','SRC_SUBJECT_ID','INTERVIEW_DATE'], axis=1).copy() 		## removing irrelevant columns
trainyValy_moredt = pd.merge(all_data[['SUBJECTKEY','INTERVIEW_AGE','sex_bin','STUDY_COHORT_NAME']], trainy.append(valy), left_on='SUBJECTKEY', right_on='subject')

# #### ^^^ Training set should have been 3739, test should have been 4515 <<< In order to use above we would need imputation

# ### Adding missing subjects (no volumetrix data) and imputing their volumetrix values

## Appending missing Subjects

missing_dict={'test':list(set(testy.subject).difference(set(testx.SUBJECTKEY))), 'train':list(set(trainy.subject).difference(set(trainx.SUBJECTKEY)))}

for key, values in missing_dict.items():
    if key=='test':
        studyname = 'ABCD NP Challenge 2019 Test Set'
        for key in values:
            all_data = all_data.append([{'SUBJECTKEY':key, 'STUDY_COHORT_NAME':studyname}], ignore_index=True)
    else:
        studyname = 'ABCD NP Challenge 2019 Training Set'
        for key in values:
            all_data = all_data.append([{'SUBJECTKEY':key, 'STUDY_COHORT_NAME':studyname}], ignore_index=True)



## Randomly assigning sex to appended subjects 

np.random.seed(seed=234)
all_data['sex_bin'] = all_data['sex_bin'].fillna( pd.Series(np.random.choice(all_data[all_data['sex_bin'].notna()]['sex_bin'], size=len(all_data.index)))                                                ) 
print(all_data['sex_bin'].value_counts())
print(all_data['sex_bin'].isna().value_counts())


## Imputing mean for rest of the columns 

all_data = all_data.fillna(all_data.mean()).copy()


## Normalizing and checking for correlation

idv_scaler = MinMaxScaler(feature_range=(0, 1))
idv_scaler.fit(all_data[all_data.columns.difference(['SUBJECTKEY','STUDY_COHORT_NAME','sex_bin'])])
all_data[all_data.columns.difference(['SUBJECTKEY','STUDY_COHORT_NAME','sex_bin'])] =     idv_scaler.transform(all_data[all_data.columns.difference(['SUBJECTKEY','STUDY_COHORT_NAME','sex_bin'])])


## Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop_dueTOhighCor = [column for column in upper.columns if any(upper[column] > 0.95)]
all_data = all_data[all_data.columns.difference(to_drop_dueTOhighCor)].copy()

## checked for 0 variance columns, there is none.


### Splitting in train (80%), validation_internal (5%), validation_forEnsemble (15%) with strata on sex, IQ and age
## Subselecting stratas

trainyValy_moredt['age_bin']=pd.qcut(trainyValy_moredt['INTERVIEW_AGE'], 5, labels=list(range(0,5)))
trainyValy_moredt['Score_bin']=pd.qcut(trainyValy_moredt['residual_fluid_intelligence_score'], 10, labels=list(range(0,10)))

val = trainyValy_moredt[trainyValy_moredt['STUDY_COHORT_NAME']=='ABCD NP Challenge 2019 Validation Set']
trainy_moredt = trainyValy_moredt[trainyValy_moredt['STUDY_COHORT_NAME']=='ABCD NP Challenge 2019 Training Set']

## splitting training into 80-20 and the 20 further into 75-25 (75 for ensemble and 25 our internal validation)

train, vals = train_test_split(trainy_moredt, test_size=0.20, random_state=243,stratify=trainy_moredt[['Score_bin', 'sex_bin','age_bin']])
val_ForEnsemble, val_internal = train_test_split(vals, test_size=0.25, random_state=243, stratify=vals[['Score_bin', 'sex_bin','age_bin']])


## Adding category column
train['sample'] = 'train'
val_ForEnsemble['sample'] = 'val_forEnsemble'
val_internal['sample'] = 'val_internal'
val['sample'] = 'val'


## Saving data with fluid scores
allY_withsegments = train.append(val_ForEnsemble).append(val_internal).append(val).copy()
allY_withsegments.to_csv(data_dir+'allY_withsegments.csv', sep=',',index=False)


## Saving clean and normalized ROI volume data
all_data = pd.merge(all_data, allY_withsegments[['residual_fluid_intelligence_score','SUBJECTKEY','sample']], on='SUBJECTKEY',how='left')
all_data.to_csv(data_dir+'all_volumetrix_processed.csv', sep=',',index=False)
