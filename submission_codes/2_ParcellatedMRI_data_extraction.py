####################### ReadMe ########################
#
# 1) The grey matter parecllated images are used to gather two kinds of datatypes corresponding to each region from SRI 24 atlas
#       a. Count of voxels in each region (amount of grey matter in that region)
#       b. Mean intensity of voxels in each region 
# 2) A slightly modified version of nilearn module's NiftiExtractor is used which is imported via utils_my_nilearn
# 
####################### ------ ########################



import numpy as np
import os
import time
import pandas as pd
import os
import utils_my_nilearn
from utils_my_nilearn import *
import sys


## Define paths

# path = '/home/...'  # Give parent path

data_dir = os.path.join(os.sep, path) 

all_Y = pd.read_csv(os.path.join(data_dir,"all_volumetrix_processed.csv"), index_col=False)

train_name = 'ABCD NP Challenge 2019 Training Set'
test_name = 'ABCD NP Challenge 2019 Test Set'
val_name = 'ABCD NP Challenge 2019 Validation Set'

names = list([train_name,test_name,val_name])
for todo in names:
    df_todo = all_Y[all_Y['STUDY_COHORT_NAME']==todo]
    if 'Train' in todo:
        folder_name = 'train'
    elif 'Test' in todo :
        folder_name = 'test'
    else :
        folder_name = 'val'

    extractor = Extractor(folder_name,n_jobs=8,data_dir=data_dir)
    count,intensity = extractor.extract_atlasdata(df_todo,folder_name,n_jobs=8)
    count["ID"] = count.index
    intensity["ID"] = intensity.index

    df_name1,df_name2 = folder_name+'_count_grey2.csv',folder_name+'_intensity_grey2.csv'

    count.to_csv(os.path.join(os.sep, data_dir,df_name1), sep=',',index=True)
    intensity.to_csv(os.path.join(os.sep, data_dir,df_name2), sep=',',index=True)

