####################### ReadMe ########################
#
# Ensemble selection method based on Caruana paper
#
####################### ------ ########################


## Define paths

# path = '/home/...'  # Give path to folder where all scores are saved

setwd(path)

dir.create("ensemble_results")



## MSE
mse <- function (obs, pred)
{
  out <- array()
  for(i in 1:ncol(pred)){
  	out[i] <- mean((obs - pred[,i])^2)
    }
  as.numeric(out)
}


ense_util <- function(data){
	data=data
	col_ordered = c('SUBJECTKEY','target',setdiff(colnames(data),c('SUBJECTKEY','target')))
	data = data[,col_ordered]
	model_names = setdiff(colnames(data),c('SUBJECTKEY'))
	EnSourceGrid=data[,model_names]
	options <- ncol(EnSourceGrid) - 1  # FIX
	EnworkingGrid <- as.matrix(EnSourceGrid[,2:(options + 1)])
	score_diagonal_vo <- diag(options)
	scoreVector_vo <- EnworkingGrid %*% score_diagonal_vo
	scores_vo <- mse(EnSourceGrid[,1],scoreVector_vo)
	list(EnSourceGrid=EnSourceGrid,EnworkingGrid=EnworkingGrid,options=options,scores_vo=scores_vo)
	}


start <- Sys.time()


ensembleSource_val = "val_forEnsemble_cor_removed.csv"
ensembleSource=read.csv(ensembleSource_val,as.is=T)
val_internal = "val_internal_cor_removed.csv"
val_intSource=read.csv(val_internal,as.is=T)
val_official = "val_official_cor_removed.csv"
val_officialSource=read.csv(val_official,as.is=T)

#INITIALIZE ARRAYS

mseTemp <- array()
finalBest <- array()

# READ IN DATA
# ensembleGrid <- read.csv(ensembleSource, header=TRUE, row.names=1)

model_names = setdiff(colnames(ensembleSource),c('SUBJECTKEY'))

ensem_all = ense_util(ensembleSource)
ensembleGrid = ensem_all[[1]]
workingGrid = ensem_all[[2]]
options = ensem_all[[3]]
scores = ensem_all[[4]]


########### doing the same with other two sets ##########
val_int_all = ense_util(val_intSource)
val_intSourceGrid = val_int_all[[1]]
val_intworkingGrid = val_int_all[[2]]
scores_vi = val_int_all[[4]]

val_official_all = ense_util(val_officialSource)
val_officialSourceGrid = val_official_all[[1]]
val_officialworkingGrid = val_official_all[[2]]
scores_vo = val_official_all[[4]]

#########################################################



# Ranking Candidates
ranked=cbind(col=colnames(ensembleGrid[,2:(options + 1)]),score=scores,score_val_int=scores_vi,score_val_off=scores_vo)
ranked=as.data.frame(ranked)
ranked$score=as.numeric(as.character(ranked$score))
ranked$score_val_int=as.numeric(as.character(ranked$score_val_int))
ranked$score_val_off=as.numeric(as.character(ranked$score_val_off))
ranked=ranked[order(ranked[,2]),]
ranked=cbind(rank=c(1:nrow(ranked)),ranked)
ranked$col<-as.character(ranked$col)



#finding top 10 candidates
for(i in 1:10){
	a=which( colnames(ensembleGrid) == ranked[ranked$rank==i,2] )-1
	if (i==1){top_ranked=a}
	else{top_ranked=c(top_ranked,a)}
}



# Testing on hyper parameters of Ensemble search


set.seed(243)
# Nbags = 30
threshold = c(0.1,0.01,0.001,0.0001) # Stopping condition for MSE improvement
fraction = c(0.3,0.4,0.5,0.6,0.7) # Fraction of models in each bag
init_weight = c(1) # initial weight given to top ranking models
ensemble_parameters=expand.grid(threshold = threshold, fraction = fraction, init_weight = init_weight)
ensemble_parameters$Nbags = ifelse(ensemble_parameters$fraction<0.5,50,30)
 

for (row in 1:nrow(ensemble_parameters)){

	Nbags = ensemble_parameters[row,'Nbags']
	threshold = ensemble_parameters[row,'threshold']
	fraction = ensemble_parameters[row,'fraction']
	init_weight = ensemble_parameters[row,'init_weight']


	
				
	# INITIALIZE THE MASTER LOG
	masterLogg <- matrix(0, 1, options)


	library(doSNOW)  
	library(foreach) 

	cl <- makeCluster(8) #
	registerDoSNOW(cl)

	# FOR
	j=1
	logg_out <- foreach (i=1:Nbags) %dopar% {

		# sample the sources (bag)
		bag <- sample(0,options,replace=T)
		bag[sample(1:options,options*fraction,replace=F)]<-1


		# INITIALIZE THE LOG

		logg <- matrix(0, 1, options)  
		logg[1,top_ranked]<-init_weight ### initialize with top models

		diagonal <- diag(options)
		diagonal <- diagonal * bag
		y=0

		# WHILE
		while(TRUE) {
			start.time<-Sys.time()
			y=y+1
			tempTempLogg <- as.numeric(logg)
			tempTempLogg <- tempTempLogg / (sum(tempTempLogg))
			tempAvgVector <- workingGrid %*% tempTempLogg
			currentBest <- mse(ensembleGrid[,1],tempAvgVector)
			currentBest_vali <- mse(val_intSourceGrid[,1],tempAvgVector)  ## For internal testing
			currentBest_valo <- mse(val_officialSourceGrid[,1],tempAvgVector) ## For internal testing

			mogg <- matrix(logg, options, options, byrow = TRUE)
			mogg <- mogg + diagonal

			mogg <- mogg / rowSums(mogg)
			rrrr <- workingGrid %*% t(mogg)
			mseTemp <- mse(ensembleGrid[,1],rrrr)
			if (min(mseTemp) <= (currentBest - threshold)) {
				k <- which.min(mseTemp)
				logg[k] <- logg[k] + 1

			}else{
				break
			}
			end.time <- Sys.time() 
			Runtime=as.numeric(end.time - start.time)
				
			if (y==1){
				Runtime2=Runtime
				out=as.data.frame(cbind(Bag=i,Run=y,Nbags=Nbags,threshold=threshold,fraction=fraction,MSE= currentBest,MSE_int=currentBest_vali ,MSE_off=currentBest_valo,Runtime=Runtime,Total_time=Runtime2))
			}else{
				Runtime2=Runtime2+Runtime
				out=as.data.frame(cbind(Bag=i,Run=y,Nbags=Nbags,threshold=threshold,fraction=fraction,MSE= currentBest,MSE_int=currentBest_vali ,MSE_off=currentBest_valo,Runtime=Runtime,Total_time=Runtime2))
			}

			write.table(out, "ensemble_results/caruana_outlog_parallel_final_save.csv", sep = ",", row.names=FALSE,col.names = FALSE, append=TRUE)
		}
		print(i)

		#save each bag's final output
		logg
	}

	stopCluster(cl)


	setwd(path)
	##  make master output as sum of all
	masterLogg_back = masterLogg
	for (i in 1:Nbags){
		masterLogg = masterLogg + logg_out[[i]]
	}

	print(paste0("Finished with ",j,"/",nrow(ensemble_parameters)))
	j=j+1
	options(width=200)
	sort(masterLogg)
	colnames(ensembleGrid)[which(masterLogg>40)+1]
	ranked$row_number = as.numeric(row.names(ranked))
	ranked = ranked[order(ranked[,'row_number']),]
	ranked_check = cbind(ranked,weight = masterLogg[,])
	sum_w = sum(ranked_check$weight)
	ranked_check$weight_fraction = ranked_check$weight /sum_w

	ranked_check = ranked_check[order(-ranked_check$weight),]
	filename = paste0(path,"ensemble_results/Ensemble_cor_removed_",threshold,"_",fraction,"_save.csv")
	write.csv(ranked_check,filename,row.names=F)

	# final result
	master <- as.numeric(masterLogg)
	master <- master / (sum(master))
	tempAvgVector <- workingGrid %*% master
	ensBest <- mse(ensembleGrid[,1],tempAvgVector)
	print(paste0("Val Ensemble MSE > ",ensBest))
	## 0.1168 when 22 models


	val_intAvgVector <- val_intworkingGrid %*% master
	val_intcurrentBest <- mse(val_intSourceGrid[,1],val_intAvgVector)
	print(paste0("Val Internal MSE > ",val_intcurrentBest))

	val_officialAvgVector <- val_officialworkingGrid %*% master
	val_officialcurrentBest <- mse(val_officialSourceGrid[,1],val_officialAvgVector)
	print(paste0("Val Official MSE > ",val_officialcurrentBest))

	if (row==1){
		out2=as.data.frame(cbind(Nbags=Nbags,threshold=threshold,fraction=fraction,MSE_ens= ensBest,MSE_int=val_intcurrentBest ,MSE_off=val_officialcurrentBest))
	}else{
		out2=as.data.frame(cbind(Nbags=Nbags,threshold=threshold,fraction=fraction,MSE_ens= ensBest,MSE_int=val_intcurrentBest ,MSE_off=val_officialcurrentBest))
		# out = rbind(out,out2)
	}

	write.table(out2,  paste0(path,"ensemble_results/caruana_scores_testing_save.csv"), sep = ",", row.names=FALSE,col.names = FALSE, append=TRUE)

}
