####################### ReadMe ########################
#
# Building candidates for ensemble via grid search method. Instead of finding the best performing candidate, I am saving predictions of every model.
#
####################### ------ ########################



import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
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

for data in data_loc:
    if data=='all_countGrey_data.csv':
        name_dt = 'count'
    elif data=='all_instensityGrey_data.csv':
        name_dt = 'intensity'
    else:
        name_dt = 'volume'

    all_x = pd.read_csv(data_dir+data, index_col=False,low_memory=False)
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
    ##SVR
    parameters_svr = {'kernel':('linear', 'rbf'), 'gamma':('auto','scale'), 'C':[0.05,0.5,1,2,5,8,10],'epsilon':[0.05,0.5,1,2,5,8,10]}
    parameters_dict_svr = expand_grid(parameters_svr)

    ##Random Forest
    parameters_rf = {'max_depth':[1,2,3,6,10], 'min_samples_split':(80,50,30,5), 'n_estimators':(30,50,100,150,500), 'max_features':('auto','sqrt','log2')}
    parameters_dict_rf = expand_grid(parameters_rf)

    ##Gradient Boosting
    parameters_gb = {'max_depth':[1,2,3,6,10], 'min_samples_split':(80,50,30,5), 'n_estimators':(30,50,100,150,500), 'max_features':('auto','sqrt','log2')}
    parameters_dict_gb = expand_grid(parameters_gb)


    def score_saver(dataframe,name,pred,name_dt):
        df_pred = pred
        name_model = df_pred.columns.values[0]
        # df_pred = pd.DataFrame(pred)
        dataframe.reset_index(drop=True, inplace=True)
        df3 = pd.concat([dataframe, df_pred], axis=1)
        # print(name+" has shape "+str(df3.shape))
        df3.columns = ['SUBJECTKEY','target',name_model]
        parent_saver = name+"_scores_"+name_dt+".csv"
        if os.path.isfile(os.path.join(data_dir,"scores",parent_saver)):
            scores = pd.read_csv(os.path.join(data_dir,"scores",parent_saver),index_col=False)
            scores = scores.merge(df3[['SUBJECTKEY',name_model]], left_on='SUBJECTKEY', right_on='SUBJECTKEY')
            scores.to_csv(os.path.join(data_dir,"scores",parent_saver), sep=',',index=False)
        else :
            df3.to_csv(os.path.join(data_dir,"scores",parent_saver), sep=',',index=False)



    def score_saver2(segment,model_name,pred):   
        df_pred = pd.DataFrame(pred)
        df_pred.columns = [model_name+'_pred']
        return df_pred



    def build_each_svr(index, row):
        # print(index)
        kernel = row['kernel']
        gamma = row['gamma']
        C = row['C']
        epsilon = row['epsilon']
        ind_val  = index
        model_name = name_dt+"_SVR_"+str(ind_val)
        reg = SVR(gamma=gamma, C=C, epsilon=epsilon)
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



    def build_each_rf(index, row): 
        max_depth = row['max_depth']
        min_samples_split = row['min_samples_split']
        n_estimators = row['n_estimators']
        max_features = row['max_features']
        ind_val  = index
        model_name = name_dt+"_RF_"+str(ind_val)
        reg = RandomForestRegressor(max_depth=max_depth,min_samples_split=min_samples_split,n_estimators=n_estimators,max_features=max_features,n_jobs=20)
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


    def build_each_gb(index, row):

        max_depth = row['max_depth']
        min_samples_split = row['min_samples_split']
        n_estimators = row['n_estimators']
        max_features = row['max_features']
        ind_val  = index
        model_name = name_dt+"_GB_"+str(ind_val)
        reg = GradientBoostingRegressor(max_depth=max_depth,min_samples_split=min_samples_split,n_estimators=n_estimators,max_features=max_features)
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



    svr_out = Parallel(n_jobs=8)(delayed(build_each_svr)(index, row) for index,row in parameters_dict_svr.iterrows())

    for i in range(len(svr_out)):
        pred_val, pred_val_ens, pred_val_int, pred_test = svr_out[i]
        score_saver(val_y,'val',pred_val,name_dt)
        score_saver(val_ens_y,'val_forEnsemble',pred_val_ens,name_dt)
        score_saver(val_int_y,'val_internal',pred_val_int,name_dt)
        score_saver(test_y,'test',pred_test,name_dt)

    rf_out = Parallel(n_jobs=8)(delayed(build_each_rf)(index, row) for index,row in parameters_dict_rf.iterrows())

    for i in range(len(rf_out)):
        pred_val, pred_val_ens, pred_val_int, pred_test = rf_out[i]
        score_saver(val_y,'val',pred_val,name_dt)
        score_saver(val_ens_y,'val_forEnsemble',pred_val_ens,name_dt)
        score_saver(val_int_y,'val_internal',pred_val_int,name_dt)
        score_saver(test_y,'test',pred_test,name_dt)



    gb_out = Parallel(n_jobs=20)(delayed(build_each_gb)(index, row) for index,row in parameters_dict_gb.iterrows())

    for i in range(len(gb_out)):
        pred_val, pred_val_ens, pred_val_int, pred_test = gb_out[i]
        score_saver(val_y,'val',pred_val,name_dt)
        score_saver(val_ens_y,'val_forEnsemble',pred_val_ens,name_dt)
        score_saver(val_int_y,'val_internal',pred_val_int,name_dt)
        score_saver(test_y,'test',pred_test,name_dt)

