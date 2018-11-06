setwd("C:/R")

ndata <- read.csv("ida_2016_test_set_update.csv", header=T, na.strings="na")
dim(ndata)
colnames(ndata)

# Drop features with high level of NAs and low predictive potential
drop <- which(colnames(ndata)=="ab_000")
ndata <- ndata[,-drop]
print("drop ab_000")
print(drop)

drop <- which(colnames(ndata)=="cr_000")
ndata <- ndata[,-drop]
print("drop cr_000")
print(drop)

drop <- which(colnames(ndata)=="ch_000")
ndata <- ndata[,-drop]
print("drop ch_000")
print(drop)

# Second group with high and 'missing together' NAs pattern
group2_labels <- c("ad_000", "cf_000", "cg_000", "co_000")
group2_na <- (is.na(ndata$ad_000))|(is.na(ndata$cf_000))|(is.na(ndata$cg_000))|(is.na(ndata$co_000))

ndata <- cbind(ndata, group2_na=ifelse(group2_na==TRUE,1,0)) # to save space when storing as csv

for (j in 1:length(group2_labels)) {
	label <- which(colnames(ndata)==group2_labels[j])
	ndata[,label] <- log(ndata[,label]+1) 			# log-transformed feature
	cases <- !is.na(ndata[,label])				# pick non-NA observations
	nzcases <- (!is.na(ndata[,label]))&(ndata[,label]!=0)	# non-zero observations
	nzeros <- sum(nzcases)					# number of non-zero cases
	total <- sum(cases)					# number of existing cases
	med <- median(ndata[nzcases,label])			# median of non-zero values
	set.seed(j)						# set a fancy seed
	impute <- rbinom(nrow(ndata),1,(nzeros/total))*med	# prepare values for imputation
	ndata[!cases,label] <- impute[!cases]			# impute where NA
	print(colnames(ndata)[label])
	print(nzeros)
	print(total)
	print(med)
}

# Third group with high and 'missing together' NAs pattern
drop <- which(colnames(ndata)=="cv_000")
ndata <- ndata[,-drop]

group3_labels <- c("ct_000", "cu_000", "cx_000", "cy_000", "cz_000", "da_000", "db_000", "dc_000")
group3_na <- (is.na(ndata$dc_000))|(is.na(ndata$db_000))|(is.na(ndata$da_000))|(is.na(ndata$cz_000))|(is.na(ndata$cy_000))|(is.na(ndata$cx_000))|(is.na(ndata$cu_000))|(is.na(ndata$ct_000))

ndata <- cbind(ndata, group3_na=ifelse(group3_na==TRUE,1,0))

for (j in 1:length(group3_labels)) {
	label <- which(colnames(ndata)==group3_labels[j])
	ndata[,label] <- log(ndata[,label]+1) 			# log-transformed feature
	cases <- !is.na(ndata[,label])				# pick non-NA observations
	nzcases <- (!is.na(ndata[,label]))&(ndata[,label]!=0)	# non-zero observations
	nzeros <- sum(nzcases)					# number of non-zero cases
	total <- sum(cases)					# number of existing cases
	med <- median(ndata[nzcases,label])			# median of non-zero values
	set.seed(j)						# set a fancy seed
	impute <- rbinom(nrow(ndata),1,(nzeros/total))*med	# prepare values for imputation
	ndata[!cases,label] <- impute[!cases]			# impute where NA
	print(colnames(ndata)[label])
	print(nzeros)
	print(total)
	print(med)
}

# Other variables with high missingness levels
suspect_labels <- c("bk_000", "bl_000", "bm_000", "bn_000", "bo_000", "bp_000", "bq_000", "br_000", "cl_000", "cm_000", "ec_00", "ed_000")

for (j in 1:length(suspect_labels)) {
	label <- which(colnames(ndata)==suspect_labels[j])
	ndata[,label] <- log(ndata[,label]+1) 			# log-transformed feature
	cases <- !is.na(ndata[,label])				# pick non-NA observations
	nzcases <- (!is.na(ndata[,label]))&(ndata[,label]!=0)	# non-zero observations
	nzeros <- sum(nzcases)					# number of non-zero cases
	total <- sum(cases)					# number of existing cases
	med <- median(ndata[nzcases,label])			# median of non-zero values
	set.seed(j)						# set a fancy seed
	impute <- rbinom(nrow(ndata),1,(nzeros/total))*med	# prepare values for imputation
	ndata[!cases,label] <- impute[!cases]			# impute where NA
	print(colnames(ndata)[label])
	print(nzeros)
	print(total)
	print(med)
}


# Treating histogram variables
attach(ndata)
ag <- cbind(ag_000, ag_001, ag_002, ag_003, ag_004, ag_005, ag_006, ag_007, ag_008, ag_009)
ay <- cbind(ay_000, ay_001, ay_002, ay_003, ay_004, ay_005, ay_006, ay_007, ay_008, ay_009)
az <- cbind(az_000, az_001, az_002, az_003, az_004, az_005, az_006, az_007, az_008, az_009)
ba <- cbind(ba_000, ba_001, ba_002, ba_003, ba_004, ba_005, ba_006, ba_007, ba_008, ba_009)
cn <- cbind(cn_000, cn_001, cn_002, cn_003, cn_004, cn_005, cn_006, cn_007, cn_008, cn_009)
cs <- cbind(cs_000, cs_001, cs_002, cs_003, cs_004, cs_005, cs_006, cs_007, cs_008, cs_009)
ee <- cbind(ee_000, ee_001, ee_002, ee_003, ee_004, ee_005, ee_006, ee_007, ee_008, ee_009)
detach(ndata)

ag <- log(ag+1)
ay <- log(ay+1)
az <- log(az+1)
ba <- log(ba+1)
cn <- log(cn+1)
cs <- log(cs+1)
ee <- log(ee+1)

hist_na <- (is.na(ag[,1]))|(is.na(ay[,1]))|(is.na(az[,1]))|(is.na(ba[,1]))|(is.na(cn[,1]))|(is.na(cs[,1]))|(is.na(ee[,1]))
ndata <- cbind(ndata, hist_na=ifelse(hist_na==TRUE,1,0))

# means for alternative feature and median value for NAs
ag_m <- apply(ag,1,mean); ag_m[is.na(ag_m)] <- median(ag_m, na.rm=T)
ay_m <- apply(ay,1,mean); ay_m[is.na(ay_m)] <- median(ay_m, na.rm=T)
az_m <- apply(az,1,mean); az_m[is.na(az_m)] <- median(az_m, na.rm=T)
ba_m <- apply(ba,1,mean); ba_m[is.na(ba_m)] <- median(ba_m, na.rm=T)
cn_m <- apply(cn,1,mean); cn_m[is.na(cn_m)] <- median(cn_m, na.rm=T)
cs_m <- apply(cs,1,mean); cs_m[is.na(cs_m)] <- median(cs_m, na.rm=T)
ee_m <- apply(ee,1,mean); ee_m[is.na(ee_m)] <- median(ee_m, na.rm=T)

# argmax for alternative feature, zeros for NAs
ag_f <- rep(0,nrow(ndata))
ay_f <- rep(0,nrow(ndata))
az_f <- rep(0,nrow(ndata))
ba_f <- rep(0,nrow(ndata))
cn_f <- rep(0,nrow(ndata))
cs_f <- rep(0,nrow(ndata))
ee_f <- rep(0,nrow(ndata))

for(i in 1:nrow(ndata)){
	if (hist_na[i]) {
		# remain zero for NAs		
	}else{
		ag_f[i] <- which.max(ag[i,])
		ay_f[i] <- which.max(ay[i,])
		az_f[i] <- which.max(az[i,])
		ba_f[i] <- which.max(ba[i,])
		cn_f[i] <- which.max(cn[i,])
		cs_f[i] <- which.max(cs[i,])
		ee_f[i] <- which.max(ee[i,])
	}
}

ndata <- cbind(ndata, ag_m, ay_m, az_m, ba_m, cn_m, cs_m, ee_m, ag_f, ay_f, az_f, ba_f, cn_f, cs_f, ee_f)

# Imputing median distribution (median for each bin) for NAs
mag <- apply(ag, 2, median, na.rm=T)
may <- apply(ay, 2, median, na.rm=T)
maz <- apply(az, 2, median, na.rm=T)
mba <- apply(ba, 2, median, na.rm=T)
mcn <- apply(cn, 2, median, na.rm=T)
mcs <- apply(cs, 2, median, na.rm=T)
mee <- apply(ee, 2, median, na.rm=T)
for (i in 1:10){
	ag[is.na(ag[,i]),i] <- mag[i]
	ay[is.na(ay[,i]),i] <- may[i]
	az[is.na(az[,i]),i] <- maz[i]
	ba[is.na(ba[,i]),i] <- mba[i]
	cn[is.na(cn[,i]),i] <- mcn[i]
	cs[is.na(cs[,i]),i] <- mcs[i]
	ee[is.na(ee[,i]),i] <- mee[i]
}

# Amending ndata for histograms
j <- which(colnames(ndata)=="ag_000"); ndata[,j:(j+9)] <- ag
j <- which(colnames(ndata)=="ay_000"); ndata[,j:(j+9)] <- ay
j <- which(colnames(ndata)=="az_000"); ndata[,j:(j+9)] <- az
j <- which(colnames(ndata)=="ba_000"); ndata[,j:(j+9)] <- ba
j <- which(colnames(ndata)=="cn_000"); ndata[,j:(j+9)] <- cn
j <- which(colnames(ndata)=="cs_000"); ndata[,j:(j+9)] <- cs
j <- which(colnames(ndata)=="ee_000"); ndata[,j:(j+9)] <- ee


# Dealing with all the rest "non-suspicious" features

# Working out the labels list for all other features
hist_flag <- rep(FALSE, ncols)
hist_labels <- c("ag_000", "ag_001", "ag_002", "ag_003", "ag_004", "ag_005", "ag_006", "ag_007", "ag_008", "ag_009", "ay_000", "ay_001", "ay_002", "ay_003", "ay_004", "ay_005", "ay_006", "ay_007", "ay_008", "ay_009", "az_000", "az_001", "az_002", "az_003", "az_004", "az_005", "az_006", "az_007", "az_008", "az_009", "ba_000", "ba_001", "ba_002", "ba_003", "ba_004", "ba_005", "ba_006", "ba_007", "ba_008", "ba_009", "cn_000", "cn_001", "cn_002", "cn_003", "cn_004", "cn_005", "cn_006", "cn_007", "cn_008", "cn_009", "cs_000", "cs_001", "cs_002", "cs_003", "cs_004", "cs_005", "cs_006", "cs_007", "cs_008", "cs_009", "ee_000", "ee_001", "ee_002", "ee_003", "ee_004", "ee_005", "ee_006", "ee_007", "ee_008", "ee_009")


others_flag <- rep(TRUE, ncol(ndata))
omit_labels <- append(group2_labels, group3_labels)
omit_labels <- append(omit_labels, suspect_labels)
omit_labels <- append(omit_labels, hist_labels)
omit_labels
for(j in 1:length(omit_labels)){
	i <- which(colnames(ndata)==omit_labels[j])
	others_flag[i] <- FALSE	
}

others_flag[1:2] <- FALSE # that's the ID and class variable, not a feature

sum(others_flag)
other_labels <- colnames(ndata)[others_flag]

# Labels of the features to deal with
other_labels

ac_na <- ifelse(!is.na(ndata$ac_000),1,0)

group2500_na <- ifelse((is.na(ndata$ac_000))|(is.na(ndata$af_000))|(is.na(ndata$av_000))|(is.na(ndata$ax_000))|(is.na(ndata$be_000))|(is.na(ndata$bf_000))|(is.na(ndata$ce_000))|(is.na(ndata$dd_000)),1,0)

group2723_na <- ifelse((is.na(ndata$ar_000))|(is.na(ndata$bc_000))|(is.na(ndata$bd_000))|(is.na(ndata$bz_000))|(is.na(ndata$cp_000))|(is.na(ndata$de_000))|(is.na(ndata$dp_000))|(is.na(ndata$dq_000))|(is.na(ndata$dr_000))|(is.na(ndata$ds_000))|(is.na(ndata$dt_000))|(is.na(ndata$du_000))|(is.na(ndata$dv_000))|(is.na(ndata$dx_000))|(is.na(ndata$dy_000)), 1, 0)

group4800_na <- ifelse((is.na(ndata$df_000))|(is.na(ndata$dg_000))|(is.na(ndata$dh_000))|(is.na(ndata$di_000))|(is.na(ndata$eb_000)),1,0)

drop_labels <- c("ah_000", "al_000", "ao_000", "aq_000", "as_000", "au_000","bb_000", "bg_000", "bt_000", "bv_000", "bx_000", "ci_000", "dj_000", "dk_000", "dl_000", "dm_000", "dz_000", "cd_000", "cq_000", "dn_000", "ds_000", "ea_000", "ef_000", "eg_000")

for(i in 1:length(drop_labels)){
	drop <- which(colnames(ndata)==drop_labels[i])
	ndata <- ndata[,-drop]
	others_flag <- others_flag[-drop]	
}

ndata <- cbind(ndata, ac_na, group2500_na, group2723_na, group4800_na)
others_flag <- append(others_flag, rep(TRUE,4))

colnames(ndata)
# 21 additional features at the end of the dataset
# log-transform original values

idata <- ndata[,others_flag] 
idata[, 1:((ncol(idata)-20))] <- log(idata[,1:(ncol(idata)-20)]+1)
summary(idata)

library(mice)
tempdata <- mice(idata, m=1, maxit=3, meth='pmm', seed=1)
idata <- complete(tempdata,1) # imputed 1 dataset, though more can be done

ndata[,others_flag] <- idata
summary(ndata)

# Chose to drop due to multicollinearity or uselessness (ID variable) several more features:
drop_labels <- c("af_000", "an_000", "ap_000", "bh_000", "bj_000", "cc_000", "ct_000", "ac_na")
for(j in 1:length(drop_labels)){
	drop <- which(colnames(ndata)==drop_labels[j])
	ndata <- ndata[,-drop]
}
dim(ndata)

#############################################################################################
# write the dataset into a file so that there is no need to do all these steps every time

write.csv(ndata, "use_me_test_data.csv")
