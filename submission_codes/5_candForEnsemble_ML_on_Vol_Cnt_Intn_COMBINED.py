####################### ReadMe ########################
# Combining volume,count and intensity data together.
# Building candidates for ensemble via grid search method. Instead of finding the best performing candidate, I am saving predictions of every model.
#
####################### ------ ########################



import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import mean_squared_error
import random

from itertools import product
import pandas as pd
import os
from joblib import Parallel, delayed
import joblib
import multiprocessing
import glob


## Define paths

# path = '/home/...'  # Give parent path

data_dir = os.path.join(os.sep, path)  

data_loc = ['all_countGrey_data.csv','all_instensityGrey_data.csv','all_volumetrix_processed.csv']
all_x1 = pd.read_csv(data_dir+data_loc[0], index_col=False,low_memory=False)
all_x2 = pd.read_csv(data_dir+data_loc[1], index_col=False,low_memory=False)
all_x3 = pd.read_csv(data_dir+data_loc[2], index_col=False,low_memory=False)
cols_to_use2 = all_x2.columns.difference(all_x1.columns).append(pd.Index(['SUBJECTKEY']))
all_x = all_x1.merge(all_x2[cols_to_use2], left_on='SUBJECTKEY', right_on='SUBJECTKEY')
cols_to_use3 = all_x3.columns.difference(all_x.columns).append(pd.Index(['SUBJECTKEY']))
all_x = all_x.merge(all_x3[cols_to_use3], left_on='SUBJECTKEY', right_on='SUBJECTKEY')



train_x= all_x[all_x['sample']=='train']
val_int_x= all_x[all_x['sample']=='val_internal']
val_ens_x= all_x[all_x['sample']=='val_forEnsemble']
val_x= all_x[all_x['sample']=='val']
test_x = all_x[all_x['STUDY_COHORT_NAME']=='ABCD NP Challenge 2019 Test Set']


### Brute forcing over-sampling from extreme edges of the distribution curve 
train_x['Score_bin2']=pd.qcut(abs(train_x['residual_fluid_intelligence_score']), 10,labels=list(range(0,10)))
trainY_new = train_x[train_x.Score_bin2==9].sample(500,replace=True,random_state=243)
trainY_new = trainY_new.append(train_x[train_x.Score_bin2==8].sample(700,replace=True,random_state=243))
trainY_new = trainY_new.append(train_x[train_x.Score_bin2==7].sample(500,replace=True,random_state=243))
trainY_new = trainY_new.append(train_x[train_x.Score_bin2==6].sample(500,replace=True,random_state=243))
trainY_new = trainY_new.append(train_x[train_x.Score_bin2==5].sample(500,replace=True,random_state=243))
trainY_new = trainY_new.append(train_x[train_x.Score_bin2==4].sample(200,replace=True,random_state=243))
trainY_new = trainY_new.append(train_x[train_x.Score_bin2==3].sample(100,replace=True,random_state=243))
trainY_new = trainY_new.append(train_x[train_x.Score_bin2==2].sample(100,replace=True,random_state=243))
trainY_new = trainY_new.append(train_x[train_x.Score_bin2==1].sample(100,replace=True,random_state=243))
trainY_new = trainY_new.append(train_x[train_x.Score_bin2==0].sample(100,replace=True,random_state=243))



cols_asIDV = train_x.columns.difference(['SUBJECTKEY','STUDY_COHORT_NAME','sample','residual_fluid_intelligence_score','Score_bin2'])
cols_forDV = ['SUBJECTKEY','residual_fluid_intelligence_score']
train_y =  trainY_new[cols_forDV]
train_x = trainY_new[cols_asIDV]
val_y = val_x[cols_forDV]
val_x = val_x[cols_asIDV]
val_int_y = val_int_x[cols_forDV]
val_int_x = val_int_x[cols_asIDV]
val_ens_y = val_ens_x[cols_forDV]
val_ens_x = val_ens_x[cols_asIDV]
test_y = test_x[cols_forDV]
test_x = test_x[cols_asIDV]


def expand_grid(dictionary):
   return pd.DataFrame([row for row in product(*dictionary.values())], 
                       columns=dictionary.keys())

parameters_xgb = {'max_depth':[1,2,3,6,10,20,30], 'learning_rate':(0.1,0.3,0.6), 'n_estimators':(30,50,100,150,200,500,800),'booster':('gbtree','gblinear'), 'num_parallel_tree':(1,2,3),'colsample_bytree':(1,0.5,0.1), 'colsample_bylevel':(1,0.5,0.1), 'colsample_bynode':(1,0.5,0.1)}
parameters_dict_xgb = expand_grid(parameters_xgb)
### ^^^ too many models. stopped above after 486 models
parameters_dict_xgb = parameters_dict_xgb[0:487]


def score_saver(dataframe,name,pred):
    df_pred = pred
    name_model = df_pred.columns.values[0]
    # df_pred = pd.DataFrame(pred)
    dataframe.reset_index(drop=True, inplace=True)
    df3 = pd.concat([dataframe, df_pred], axis=1)
    # print(name+" has shape "+str(df3.shape))
    df3.columns = ['SUBJECTKEY','target',name_model]
    parent_saver = name+"_scores_AllData.csv"
    if os.path.isfile(os.path.join(data_dir,"scores",parent_saver)):
        scores = pd.read_csv(os.path.join(data_dir,"scores",parent_saver),index_col=False)
        scores = scores.merge(df3[['SUBJECTKEY',name_model]], left_on='SUBJECTKEY', right_on='SUBJECTKEY')
        scores.to_csv(os.path.join(data_dir,"scores",parent_saver), sep=',',index=False)
    else :
        df3.to_csv(os.path.join(data_dir,"scores",parent_saver), sep=',',index=False)



def score_saver2(segment,model_name,pred):   
    df_pred = pd.DataFrame(pred)
    df_pred.columns = [model_name+'_'+segment+'_pred']
    return df_pred

# In[36]:

name_dt='all_data'
def build_each_xgb(index, row):

    max_depth = row['max_depth']
    learning_rate = row['learning_rate']
    n_estimators = row['n_estimators']
    booster = row['booster']
    # num_parallel_tree = row['num_parallel_tree']
    colsample_bytree = row['colsample_bytree']
    colsample_bylevel = row['colsample_bylevel']
    colsample_bynode = row['colsample_bynode']

    ind_val  = index
    model_name = "all_data_xgb_"+str(ind_val)

    reg = XGBRegressor(max_depth=max_depth
                    , learning_rate=learning_rate
                    , n_estimators=n_estimators
                    , booster=booster
                    # , num_parallel_tree=num_parallel_tree
                    , colsample_bytree=colsample_bytree
                    , colsample_bylevel=colsample_bylevel
                    , colsample_bynode=colsample_bynode
                    , n_jobs=20
                    , random_state=243)
    y = train_y['residual_fluid_intelligence_score'].values
    reg.fit(train_x, y)
    
    pred_val = reg.predict(val_x)
    pv = score_saver2('val',model_name,pred_val)
    
    pred_val_ens = reg.predict(val_ens_x)
    pve = score_saver2('val_forEnsemble',model_name,pred_val_ens)
    
    pred_val_int = reg.predict(val_int_x)
    pvi = score_saver2('val_internal',model_name,pred_val_int)
    
    pred_test = reg.predict(test_x)
    pt = score_saver2('test',model_name,pred_test)
    print(model_name," completed")
    return pv, pve, pvi, pt




xg_out = Parallel(n_jobs=20)(delayed(build_each_xgb)(index, row) for index,row in parameters_dict_xgb.iterrows())

for i in range(len(xg_out)):
    pred_val, pred_val_ens, pred_val_int, pred_test = xg_out[i]
    score_saver(val_y,'val',pred_val)
    score_saver(val_ens_y,'val_forEnsemble',pred_val_ens)
    score_saver(val_int_y,'val_internal',pred_val_int)
    score_saver(test_y,'test',pred_test)
## changing the column names of previous run
for type_dt in ['val','val_forEnsemble','val_internal','test']:
    a = pd.read_csv(data_dir+'scores/'+type_dt+'_scores_AllData.csv', index_col=False,low_memory=False)
    change = list(x+'_old' for x in a.columns[2:].values.tolist())
    not_change= list(x for x in a.columns[:2].values.tolist())
    not_change.extend(change)
    a.columns = not_change
    a.to_csv(data_dir+'scores/'+type_dt+'_scores_AllData.csv', sep=',',index=False)




### shortening down the grid
parameters_xgb2 = {'max_depth':[1,2,3,6,10,20,30]
                , 'learning_rate':(0.1,0.3,0.6)
                , 'n_estimators':(30,50,100,150,500)
                , 'booster':('gbtree','gblinear')
                , 'colsample_bytree':(1,0.5,0.1)
                , 'colsample_bylevel':(1,0.5,0.1)
                , 'colsample_bynode':(1,0.5,0.1)}

parameters_dict_xgb2 = expand_grid(parameters_xgb2)


xg_out = Parallel(n_jobs=20)(delayed(build_each_xgb)(index, row) for index,row in parameters_dict_xgb2[1:3000].iterrows())
### ^^^ too many models. stopped above after 3000 models

for i in range(len(xg_out)):
    pred_val, pred_val_ens, pred_val_int, pred_test = xg_out[i]
    score_saver(val_y,'val',pred_val)
    score_saver(val_ens_y,'val_forEnsemble',pred_val_ens)
    score_saver(val_int_y,'val_internal',pred_val_int)
    score_saver(test_y,'test',pred_test)


