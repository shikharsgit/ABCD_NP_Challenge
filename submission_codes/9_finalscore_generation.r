####################### ReadMe ########################
#
# Ensemble selection method based on Caruana paper
#
####################### ------ ########################
## Define paths

# path = '/home/...'  # Give path to folder where all scores are saved

setwd(path)

## MSE
mse <- function (obs, pred)
{
  out <- array()
  for(i in 1:ncol(pred)){
  	out[i] <- mean((obs - pred[,i])^2)
    }
  as.numeric(out)
}

#### Loading test set scores ####

myFiles <- list.files(pattern="*csv")

merge.all <- function(x, y) {
    merge(x, y[,setdiff(colnames(y),'target')], all=TRUE, by=c("SUBJECTKEY"))
}

#loading and cleaning
library(plyr)
import_test <- llply(myFiles[grep("test*",myFiles)], read.csv)
test <- Reduce(merge.all, import_test)
colnames(test)[grep("*target*",colnames(test))]
colnames(test)[grep('_test_p',colnames(test))] <- gsub('_test_p','_p',colnames(test)[grep('_test_p',colnames(test))]) 
test = test[ -grep( "\\.", names( test))] 


setwd(paste0(path,'/ensemble_results'))

myFiles <- myFiles[grep('save',myFiles)]

score_check = read.csv('caruana_scores_testing_save.csv',as.is=T,header=F)
colnames(score_check)<-c('nbag','threshold','fract','mse_ens','mse_int','mse_off')
score_check$sum_val = score_check$mse_int+score_check$mse_off

score_check=score_check[order(score_check$sum_val),]
options(scipen=999)
thres = score_check[1,'threshold']
frac = score_check[1,'fract']

## Selecting best output
top_model = read.csv(paste0("Ensemble_cor_removed_",thres,"_",frac,"_save.csv"),as.is=T)
top_model = subset(top_model,weight>0)
submodel_list = top_model$col


setwd(path)

test_submit = test[,c('SUBJECTKEY','prediction_fluid_score')] 
colnames(test_submit) <- c('subject','predicted_score')

write.csv(test_submit,'test_predictions.csv',row.names=F)