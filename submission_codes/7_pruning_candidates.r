####################### ReadMe ########################
#
# R code for pruning the library of candidates
#
####################### ------ ########################


## Define paths

# path = '/home/...'  # Give path to folder where all scores are saved

setwd(path)

myFiles <- list.files(pattern="*csv")


merge.all <- function(x, y) {
    merge(x, y[,setdiff(colnames(y),'target')], all=TRUE, by=c("SUBJECTKEY"))
}

#loading and cleaning
library(plyr)
import_val <- llply(myFiles[grep("val_score*",myFiles)], read.csv)

val_official <- Reduce(merge.all, import_val)
colnames(val_official)[grep("*target.*",colnames(val_official))]
colnames(val_official)[grep('_val_p',colnames(val_official))] <- gsub('_val_p','_p',colnames(val_official)[grep('_val_p',colnames(val_official))]) 
val_official = val_official[ -grep( "\\.", names( val_official))] 

import_valforEnsemble <- llply(myFiles[grep("val_forEns*",myFiles)], read.csv)
val_forEnsemble <- Reduce(merge.all, import_valforEnsemble)
colnames(val_forEnsemble)[grep("*target*",colnames(val_forEnsemble))]
colnames(val_forEnsemble)[grep('_val_forEnsemble_p',colnames(val_forEnsemble))] <- gsub('_val_forEnsemble_p','_p',colnames(val_forEnsemble)[grep('_val_forEnsemble_p',colnames(val_forEnsemble))]) 
val_forEnsemble = val_forEnsemble[ -grep( "\\.", names( val_forEnsemble))] 

# library(plyr)
import_val_internal <- llply(myFiles[grep("val_internal*",myFiles)], read.csv)
val_internal <- Reduce(merge.all, import_val_internal)
colnames(val_internal)[grep("*target*",colnames(val_internal))]
colnames(val_internal)[grep('_val_internal_p',colnames(val_internal))] <- gsub('_val_internal_p','_p',colnames(val_internal)[grep('_val_internal_p',colnames(val_internal))]) 
val_internal = val_internal[ -grep( "\\.", names( val_internal))] 





mse <- function (obs, pred)
{
  out <- array()
  for(i in 1:ncol(pred)){
  	out[i] <- mean((obs - pred[,i])^2)
    }
  as.numeric(out)
}


## Making a hard cut on all candidates which fails MSE for val (official validation) with >=77
col_ordered = c('SUBJECTKEY','target',setdiff(colnames(val_official),c('SUBJECTKEY','target')))
val_official = val_official[,col_ordered]
val_officialSource = val_official
model_names = setdiff(colnames(val_officialSource),c('SUBJECTKEY'))
val_officialSourceGrid=val_officialSource[,model_names]
options <- ncol(val_officialSourceGrid) - 1  # FIX
val_officialworkingGrid <- as.matrix(val_officialSourceGrid[,2:(options + 1)])
score_diagonal_vo <- diag(options)
scoreVector_vo <- val_officialworkingGrid %*% score_diagonal_vo
scores_vo <- mse(val_officialSourceGrid[,1],scoreVector_vo)


# Ranking Candidates
ranked=cbind(col=colnames(val_officialSourceGrid[,2:(options + 1)]),score=scores_vo)
ranked=as.data.frame(ranked)
ranked$score=as.numeric(as.character(ranked$score))
ranked=ranked[order(ranked[,2]),]
ranked=cbind(rank=c(1:nrow(ranked)),ranked)
ranked$col<-as.character(ranked$col)
ranked_filtered = subset(ranked,score<77 & score>30)


cols_touse = ranked_filtered$col
cols_touse = c('SUBJECTKEY','target',cols_touse)
val_forEnsemble = val_forEnsemble[,intersect(colnames(val_forEnsemble),cols_touse)]
val_official = val_official[,colnames(val_forEnsemble)]
val_internal = val_internal[,colnames(val_forEnsemble)]


val_forEnsemble_check=val_forEnsemble

library('caret')
nzv=nearZeroVar(val_forEnsemble_check)
if (length(nzv)>0) val_forEnsemble_check=val_forEnsemble_check[,-nzv]


df2 = cor(val_forEnsemble_check[,2:ncol(val_forEnsemble_check)])
df2[!lower.tri(df2)] = 0

#Taking only those models which are highly correlated with other models
a=apply(df2,2,function(x) any(x > 0.95))
a[1]=FALSE
a[length(a)+1]=FALSE
names(a)[length(a)]='SUBJECTKEY'
# a[2]=FALSE

#Removing them
val_forEnsemble_check = val_forEnsemble_check[,!a]


cols_usable = c('SUBJECTKEY','target',setdiff(colnames(val_forEnsemble_check),c('SUBJECTKEY','target')))
val_official = val_official[,cols_usable]
val_forEnsemble = val_forEnsemble[,cols_usable]
val_internal = val_internal[,cols_usable]

write.csv(val_official,paste0(path,"val_official_cor_removed.csv"),row.names=F)
write.csv(val_forEnsemble,paste0(path,"val_forEnsemble_cor_removed.csv"),row.names=F)
write.csv(val_internal,paste0(path,"val_internal_cor_removed.csv"),row.names=F)
