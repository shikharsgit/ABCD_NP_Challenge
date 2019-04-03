####################### ReadMe ########################
# 1. Create a folder SRI24 and download lpba40 atlas from SRI24 websire.
# 2. Dataloader and conv net builder are in utils_abcd
# 3. CNN models are built on particular regions around parietal and frontal cortex. More relaxed conditions for frontal regions.
# 4. Various stopping conditions are used (like trend in loss improvement or variance of scores generated)
####################### ------ ########################





import numpy as np
import os
from utils_abcd import *
import nibabel as nib
import nilearn 
from nilearn.image import load_img
import time
from time import gmtime, strftime
import pandas as pd
import glob
from math import floor
from scipy.ndimage.interpolation import zoom
import torch
from torch.utils.data import Dataset,DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose
import statistics
from itertools import chain 
import re


torch.manual_seed(243)

## Define paths

# path = '/home/...'  # Give parent path

data_dir = os.path.join(os.sep, path)  

all_Y = pd.read_csv(os.path.join(data_dir,"allY_withsegments.csv"), index_col=False)
trainY = all_Y[all_Y['sample']=='train']
valY = all_Y[all_Y['sample']=='val']
val_forEnsembleY = all_Y[all_Y['sample']=='val_forEnsemble']
val_internalY = all_Y[all_Y['sample']=='val_internal']
all_Y = pd.read_csv(os.path.join(data_dir,"all_instensityGrey_data.csv"), index_col=False)
testY = all_Y[all_Y['STUDY_COHORT_NAME']=='ABCD NP Challenge 2019 Test Set']


### Brute forcing over-sampling from extreme edges of the distribution curve 
trainY['Score_bin2']=pd.qcut(abs(trainY['residual_fluid_intelligence_score']), 10,labels=list(range(0,10)))
trainY_new = trainY[trainY.Score_bin2==9].sample(500,replace=True,random_state=243)
trainY_new = trainY_new.append(trainY[trainY.Score_bin2==8].sample(700,replace=True,random_state=243))
trainY_new = trainY_new.append(trainY[trainY.Score_bin2==7].sample(500,replace=True,random_state=243))
trainY_new = trainY_new.append(trainY[trainY.Score_bin2==6].sample(500,replace=True,random_state=243))
trainY_new = trainY_new.append(trainY[trainY.Score_bin2==5].sample(500,replace=True,random_state=243))
trainY_new = trainY_new.append(trainY[trainY.Score_bin2==4].sample(200,replace=True,random_state=243))
trainY_new = trainY_new.append(trainY[trainY.Score_bin2==3].sample(100,replace=True,random_state=243))
trainY_new = trainY_new.append(trainY[trainY.Score_bin2==2].sample(100,replace=True,random_state=243))
trainY_new = trainY_new.append(trainY[trainY.Score_bin2==1].sample(100,replace=True,random_state=243))
trainY_new = trainY_new.append(trainY[trainY.Score_bin2==0].sample(100,replace=True,random_state=243))




### Two atlases available > tzo116plus and lpba40, I used lpba 40
atlas_location = 'sri24/lpba40.nii'
labels_loc = 'sri24/LPBA40-labels.txt'


mask =  load_img(os.path.join(data_dir,atlas_location))
mask = np.squeeze(mask.get_data())
labels = pd.read_csv(os.path.join(data_dir,labels_loc))
npad = ((0, 0), (0,0), (45, 40))   ### Mask data is smaller at z axis (used MRICron to check this)
mask_padded = np.pad(mask, pad_width=npad, mode='constant', constant_values=0)
region_counts = (pd.Series(mask_padded.flatten()).value_counts())

def labelsCombine(x):
    if "L_" in x or "R_" in x:
        len_full = len(x) 
        a = x[2:len_full]
    else:
        a=x
    return a

labels["LR_combined"] = list(labelsCombine(i) for i in labels["background"])  

def organlist(x,df_p):
    df = df_p["background"]
    list_out = list(df_p.loc[df[df.str.contains(x, regex=False)].index,"0"].values) 
    return list_out


unique_labels=dict((i,organlist(i,labels)) for i in list(np.unique(labels["LR_combined"].values)))
unique_labels['cuneus'] = [67,68]


#### Taking gyruses pertaining to frontal and parietal lobes
my_keys = ['gyrus_rectus','hippocampus','inferior_frontal_gyrus','middle_frontal_gyrus','postcentral_gyrus','precentral_gyrus','precuneus','superior_frontal_gyrus','supramarginal_gyrus']
subselect_labels = {x:unique_labels[x] for x in my_keys}

param_grid = 'parameters_grid_search_CNN.csv' # provided with submission codes 
param_df = pd.read_csv(os.path.join(data_dir,param_grid))


for key, values in subselect_labels.items():
    i_my += 1
    mask_padded_modified = mask_padded.copy()

    rn_list = values        # list of anatomical parts (L+R hemisphere)
    anatomy_part = key      # name of parent anatomical part
    

    mask_padded_modified[np.isin(mask_padded_modified,rn_list,invert=True)]=0

    # argwhere will give you the coordinates of every non-zero point
    non_zero_points = np.argwhere(mask_padded_modified)
    # find the inner vertex of cube
    inner_vertex = non_zero_points.min(axis=0)
    # find the outer vertex of cuber
    outer_vertex = non_zero_points.max(axis=0)

    zoom=1

    shape0=inner_vertex-1
    shape0 = [floor(i * zoom) for i in shape0]
    shape1=outer_vertex+1
    shape1 = [floor(i * zoom) for i in shape1]
    final_shape = [a_i - b_i for a_i, b_i in zip(shape1, shape0)]
    # print(final_shape)


    def pool_reduce(x,pool_size=5,stride=2):
        y=floor((x-pool_size)/stride)+1
        return y

    def ll_neurons(shape_list,countOf_pooling_layers,pool_size=5,stride=2):
        n_req = final_shape
        for i in range(countOf_pooling_layers):
            n_req = [pool_reduce(k,pool_size) for k in n_req]
            a = n_req[0]*n_req[1]*n_req[2]
        return a, print(n_req[0]," * ",n_req[1]," * ",n_req[2])



    def my_score_saver(data_dir,segment,ids,real,pred,check_pt_name):
        df_ID = pd.DataFrame(ids)
        df_real = pd.DataFrame(real)
        df_pred = pd.DataFrame(pred)
        df3 = pd.concat([df_ID,df_real, df_pred], axis=1)
        df3.columns = ['SUBJECTKEY','target',check_pt_name+'_pred']
        df3.to_csv(os.path.join(data_dir,"scores",segment,str(check_pt_name+'_pred.csv')), sep=',',index=False)

        parent_saver = segment+"_scores.csv"
        if os.path.isfile(os.path.join(data_dir,"scores",parent_saver)):
            scores = pd.read_csv(os.path.join(data_dir,"scores",parent_saver),index_col=False)
            scores = scores.merge(df3[['SUBJECTKEY',check_pt_name+'_pred']], left_on='SUBJECTKEY', right_on='SUBJECTKEY')
            scores.to_csv(os.path.join(data_dir,"scores",parent_saver), sep=',',index=False)
        else :
            df3.to_csv(os.path.join(data_dir,"scores",parent_saver), sep=',',index=False)

    def var_wrapper(inputs,labels): 
        if isinstance(inputs, list):
            inputs = [Variable(inp.to(device)) for inp in inputs]
        else:
            inputs = Variable(inputs.to(device))
        if isinstance(labels, list):
            labels = [Variable(label.to(device)) for label in labels]
        else:
            labels = Variable(labels.to(device))

        return inputs,labels

    def model_scorer(loader,loss_all):
        out_pred=[]
        out_real=[]
        ids_all=[]
        for i, data in enumerate(loader):
            inputs, labels,ids = data['image'], data['target'], data['ID']
            inputs,labels = var_wrapper(inputs,labels)
            outputs = net(inputs)
            out_pred.extend(list(chain.from_iterable(outputs.tolist())))
            out_real.extend(list(chain.from_iterable(labels.tolist())))
            ids_all.extend(ids)
            loss = criterion(outputs, labels)
            loss_all += loss.item()

        return  ids_all, out_pred, out_real, loss_all

    def trend(data):
        trend_each = [b - a for a, b in zip(data[::1], data[1::1])]
        mean_trend = sum(trend_each)/len(trend_each)
        return mean_trend

    n_req_out = ll_neurons(final_shape,2,5,2)[0]

    for index,row in param_df.iterrows():
        # index,row = 0,param_df.loc[0]

        lr = row['lr']
        b = row['batch_size']
        nlayer1 = row['nblocklayer1'].astype(int)
        nlayer2 = row['nblocklayer2'].astype(int)
        nlayer3 = row['nblocklayer3'].astype(int)
        ind_val  = index

        anat_model_check = anatomy_part+"_"+str(ind_val)

        if (anat_model_check in done_list) or (ind_val==2 or ind_val==5):
            print("Passing on ",anat_model_check)
            continue
        else:
            pass

        if "aaaa" in anatomy_part:
            n_epochs = 100 #### harder training on certain anatomical parts
        else: 
            n_epochs = 50 


        anat_key = anatomy_part+"_"+str(i_my)+"_md_"+str(ind_val)
        date_time = strftime("%Y-%m-%d %H:%M:%S")

        print ("\n Started building ##### ",anat_key," ##### At ",date_time,"\n")

        time_start = time.time()

        def ResNet_abcd_741( **kwargs):
            nn = ResNet_abcd(BasicBlock, [nlayer1,nlayer2,nlayer3], n_req_out,**kwargs)
            return nn

        ### use multiple GPUs

        torch.cuda.empty_cache()
        gpu = 2
        multi_gpus = [2,3,4,6,7,1]
        b=int(100/4*len(multi_gpus))

        device = torch.device("cuda:" + str(gpu))
        # device = torch.device('cpu')
        # multi_gpus = None
        if multi_gpus is None:
            net = ResNet_abcd_741().to(device)
        else:
            net = torch.nn.DataParallel(ResNet_abcd_741(), device_ids=multi_gpus).cuda(gpu)

        optimizer = optim.Adam(net.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)


        criterion = nn.MSELoss().to(device)


        train_dataset = ABCD_dataset.fromcsv(trainY_new,mask_padded_modified,rn_list,os.path.join(data_dir,'train'),shape0=shape0,shape1=shape1)#,zoom_factor=zoom
        val_dataset = ABCD_dataset.fromcsv(valY,mask_padded_modified,rn_list,os.path.join(data_dir,'train'),shape0=shape0,shape1=shape1)#,zoom_factor=zoom
        val_internal_dataset = ABCD_dataset.fromcsv(val_internalY,mask_padded_modified,rn_list,os.path.join(data_dir,'train'),shape0=shape0,shape1=shape1)#,zoom_factor=zoom
        val_forEnsemble_dataset = ABCD_dataset.fromcsv(val_forEnsembleY,mask_padded_modified,rn_list,os.path.join(data_dir,'train'),shape0=shape0,shape1=shape1)#,zoom_factor=zoom
        test_dataset = ABCD_dataset.fromcsv(testY,mask_padded_modified,rn_list,os.path.join(data_dir,'train'),shape0=shape0,shape1=shape1)#,zoom_factor=zoom

        train_loader = DataLoader(train_dataset, batch_size=b, shuffle=True, num_workers=10)
        val_loader = DataLoader(val_dataset, batch_size=b, shuffle=False, num_workers=10)
        val_internal_loader = DataLoader(val_internal_dataset, batch_size=b, shuffle=False, num_workers=10)
        val_forEnsemble_loader = DataLoader(val_forEnsemble_dataset, batch_size=b, shuffle=False, num_workers=10)
        test_loader = DataLoader(test_dataset, batch_size=b, shuffle=False, num_workers=10)

        total_loss = []
        val_total_loss = []
        val_acc = []
        start_time = time.time()
        val_total_loss_new5 = []
        try:

            for epoch in range(n_epochs):
                running_loss = 0.0
                epoch_loss = 0.0
                # scheduler.step(epoch)
                # train
                net.train()
                for i, data in enumerate(train_loader):
                    inputs, labels = data['image'], data['target']
                    # wrap data in Variable
                    if isinstance(inputs, list):
                        inputs = [Variable(inp.to(device)) for inp in inputs]
                    else:
                        inputs = Variable(inputs.to(device))
                    if isinstance(labels, list):
                        labels = [Variable(label.to(device)) for label in labels]
                    else:
                        labels = Variable(labels.to(device))
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    # optimizer.backward(loss)
                    optimizer.step()
                    # print statistics
                    running_loss += loss.item()
                    epoch_loss += loss.item()


                # validate every x iterations
                segment = 'val'
                if epoch>0:
                    net.eval()
                    validation_loss = 0.0
                    total_correct = 0
                    #print("1st")
                    ids_all, out_pred, out_real, validation_loss = model_scorer(val_loader,validation_loss)
                    validation_loss /= len(val_loader)
                    #print("2nd")
                    val_total_loss.append(validation_loss)
                    my_score_saver(data_dir,segment,ids_all,out_real,out_pred,check_pt_name)        

                if epoch>0 and ((validation_loss<77 and np.var(out_pred)>0.01 and 'frontal' in anatomy_part) or (validation_loss<72 and np.var(out_pred)>0.001) ):## there were some near zero variance outputs
                    segment = 'val_internal'
                    net.eval()
                    val_internal_loss = 0.0
                    ids_all, out_pred, out_real, val_internal_loss = model_scorer(val_internal_loader,val_internal_loss)
                    my_score_saver(data_dir,segment,ids_all,out_real,out_pred,check_pt_name)        
 
                    segment = 'val_forEnsemble'
                    net.eval()
                    val_internal_loss = 0.0
                    ids_all, out_pred, out_real, val_internal_loss = model_scorer(val_forEnsemble_loader,val_internal_loss)
                    my_score_saver(data_dir,segment,ids_all,out_real,out_pred,check_pt_name)                        

                    segment = 'test'
                    net.eval()
                    val_internal_loss = 0.0
                    ids_all, out_pred, out_real, val_internal_loss = model_scorer(test_loader,val_internal_loss)
                    my_score_saver(data_dir,segment,ids_all,out_real,out_pred,check_pt_name)                        


                if epoch>6 and epoch%5==0:
                    val_total_loss_new10 = val_total_loss[(len(val_total_loss)-10):len(val_total_loss)] 
                    val_total_loss_new10_median  = statistics.median(val_total_loss_new10)
                    if epoch==10:
                        val_total_loss_best10_median=val_total_loss_new10_median
                    elif val_total_loss_new10_median<0.95*val_total_loss_best10_median:
                        val_total_loss_best10_median = val_total_loss_new10_median
                    elif (val_total_loss_new10_median<0.98*(val_total_loss_best10_median)  or abs(trend(val_total_loss_new10))>1) and trend(val_total_loss_new10)<0:
                        pass
                    else:
                        print("\nStopping building further at Epoch: ",epoch)
                        break

                error = 0

        except:
            error = 1


        date_time2 = strftime("%Y-%m-%d %H:%M:%S")

        if error==0:
            total_time = ((time.time() - time_start) / 60)
            print("\n Time trained: {:.2f} minutes".format(total_time),"\n")
            print ("\n Finshed building ##### ",anat_key," ##### At ",date_time2,"\n")
        else:

            print ("\n Error in building ##### ",anat_key," ##### At ",date_time2,"\n")




quit()
