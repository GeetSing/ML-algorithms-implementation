setwd("C:/R")

ndata <- read.csv("ida_2016_training_set_update.csv", header=T, na.strings="na")
nrows <- dim(ndata)[1]
ncols <- dim(ndata)[2]
colnames(ndata)
num_class <- ifelse(ndata$class=='neg',0,1)

summary(ndata)
# Found the 7 histogram-type variables, will have to look into them closer
# "ag_000", "ag_001", "ag_002", "ag_003", "ag_004", "ag_005", "ag_006", "ag_007", "ag_008", "ag_009" - 671na
# "ay_000", "ay_001", "ay_002", "ay_003", "ay_004", "ay_005", "ay_006", "ay_007", "ay_008", "ay_009" - 671na
# "az_000", "az_001", "az_002", "az_003", "az_004", "az_005", "az_006", "az_007", "az_008", "az_009" - 671na
# "ba_000", "ba_001", "ba_002", "ba_003", "ba_004", "ba_005", "ba_006", "ba_007", "ba_008", "ba_009" - 688na
# "cn_000", "cn_001", "cn_002", "cn_003", "cn_004", "cn_005", "cn_006", "cn_007", "cn_008", "cn_009" - 687na
# "cs_000", "cs_001", "cs_002", "cs_003", "cs_004", "cs_005", "cs_006", "cs_007", "cs_008", "cs_009" - 669na
# "ee_000", "ee_001", "ee_002", "ee_003", "ee_004", "ee_005", "ee_006", "ee_007", "ee_008", "ee_009" - 671na

# Look at the distribution of missing values
nas <- rep(0, ncols)
for (i in 1:ncols){
	nas[i] <- sum(is.na(ndata[,i]))
}
hist(nas, prob=TRUE, las=1, breaks=50, xlab="Number of missing values", main="Distribution of features by number of NAs") 
lines(density(nas), col=2)

# Look at the tail of distribution with more than 10% missing values
suspects <- colnames(ndata)[nas>0.1*nrows]
na_rate <- nas[nas>0.1*nrows]/nrows

# Check pos rates in cases feature is present and if it is missing
pos_rate <- rep(0, length(suspects))
pos_rate_na <- rep(0, length(suspects))

i <- 1
for (j in 1:ncols){
	if(nas[j]>0.1*nrows){
		cases <- complete.cases(ndata[,j])
		pos_rate[i] <- sum(ndata[cases,1]=='pos')/sum(cases)
		pos_rate_na[i] <- sum(ndata[!(cases),1]=='pos')/sum(!cases)
		i <- i+1
	}
}

print("positive rate across the dataset")
sum(ndata[,1]=="pos")/nrows
# [1] 0.016666667

cbind(suspects, na_rate, pos_rate, pos_rate_na, options(digits=3))

#      suspects na_rate pos_rate pos_rate_na  
#  [1,] "ab_000" 0.772   0.0168   0.0166      
#  [2,] "ad_000" 0.248   0.00786  0.0434      
#  [3,] "bk_000" 0.384   0.0253   0.00278     
#  [4,] "bl_000" 0.455   0.0285   0.00246     
#  [5,] "bm_000" 0.659   0.0454   0.0018      
#  [6,] "bn_000" 0.733   0.0579   0.00168     
#  [7,] "bo_000" 0.772   0.0675   0.00168     
#  [8,] "bp_000" 0.796   0.0748   0.00174     
#  [9,] "bq_000" 0.812   0.081    0.00179     
# [10,] "br_000" 0.821   0.0848   0.00183     
# [11,] "cf_000" 0.248   0.00786  0.0434      
# [12,] "cg_000" 0.248   0.00786  0.0434      
# [13,] "ch_000" 0.248   0.00786  0.0434      
# [14,] "cl_000" 0.159   0.0153   0.0239      
# [15,] "cm_000" 0.165   0.0142   0.0294      
# [16,] "co_000" 0.248   0.00786  0.0434      
# [17,] "cr_000" 0.772   0.0168   0.0166      
# [18,] "ct_000" 0.23    0.0101   0.0385      
# [19,] "cu_000" 0.23    0.0101   0.0385      
# [20,] "cv_000" 0.23    0.0101   0.0385      
# [21,] "cx_000" 0.23    0.0101   0.0385      
# [22,] "cy_000" 0.23    0.0101   0.0385      
# [23,] "cz_000" 0.23    0.0101   0.0385      
# [24,] "da_000" 0.23    0.0101   0.0385      
# [25,] "db_000" 0.23    0.0101   0.0385      
# [26,] "dc_000" 0.23    0.0101   0.0385      
# [27,] "ec_00"  0.171   0.0138   0.0307      
# [28,] "ed_000" 0.159   0.0153   0.0239      

# There're too many missing values to impute them properly, but they mostly affect pos rate, so we won't just drop them  
# After looking at the output, spotted 3 groups, that are likely to be measured in batches
# Group 1: "ab_000", "cr_000" - 77% of NAs and the same positive rate as average, drop these features
# Group 2: "ad_000", "cf_000", "cg_000", "ch_000", "co_000" - introduce dummy on whether group is NA
# Group 3: "ct_000", "cu_000", "cv_000", "cx_000", "cy_000", "cz_000", "da_000", "db_000", "dc_000" - introduce dummy on whether group is NA

drop <- which(colnames(ndata)=="ab_000")
ndata <- ndata[,-drop]
drop <- which(colnames(ndata)=="cr_000")
ndata <- ndata[,-drop]

#########################################################################################################
# Looking at Group 2: "ad_000", "cf_000", "cg_000", "ch_000", "co_000"

cases <- (!is.na(ndata$ad_000))&(ndata$ad_000<1e4) # Large outlier - look closer at smaller values
hist(ndata$ad_000[cases], prob=TRUE, las=1, breaks=50)
lines(density(ndata$ad_000[cases]), col=2) # Extremely skewed distribution, apply log(x+1)
logvar <- log(ndata$ad_000 + 1)
cases <- !is.na(ndata$ad_000)
hist(logvar[cases], prob=TRUE, las=1, breaks=100) 
lines(density(logvar[cases]), col=2)
# Looks better for further analysis: Bell shape + high probability of zero

# Other variables have the same pattern of distribution, so applied log-tranformation too
cases <- !is.na(ndata$cf_000)
logvar <- log(ndata$cf_000 + 1)
hist(logvar[cases], prob=TRUE, las=1, breaks=50)
lines(density(logvar[cases]), col=2)

cases <- !is.na(ndata$cg_000)
logvar <- log(ndata$cg_000 + 1)
hist(logvar[cases], prob=TRUE, las=1, breaks=50)
lines(density(logvar[cases]), col=2)

cases <- !is.na(ndata$co_000)
logvar <- log(ndata$co_000 + 1)
hist(logvar[cases], prob=TRUE, las=1, breaks=50)
lines(density(logvar[cases]), col=2)

summary(as.factor(ndata$ch)) # Not useful as only 10 non-zero values out of 60K observations - drop "ch_000"

#########################################################################################################
# In Group 2: "ad_000", "cf_000", "cg_000", "ch_000", "co_000"
# Drop feature "ch_000" - only 10 non-zero values, 1 out of them is pos class - not enough out of 60K observations to rely on it
# Apply log(x+1) to all non-missing values of "ad_000", "cf_000", "cg_000", "co_000"
# The variables have a one- or two-bell shape and a peak at zero
# It seems reasonable to leave transformed features even with 25% NAs
# Impute instead of NAs zero with probability of zero and median-of-non-zero with probability of non-zero
#########################################################################################################

drop <- which(colnames(ndata)=="ch_000")
ndata <- ndata[,-drop]

group2_labels <- c("ad_000", "cf_000", "cg_000", "co_000")
group2_na <- (is.na(ndata$ad_000))|(is.na(ndata$cf_000))|(is.na(ndata$cg_000))|(is.na(ndata$co_000))

ndata <- cbind(ndata, group2_na=ifelse(group2_na==TRUE,1,0))

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


pairs_data <- cbind(num_class, ndata$ad_000, ndata$cf_000, ndata$cg_000, ndata$co_000)
options(digits=3)
cor(pairs_data)
#           num_class                            
# num_class    1.0000 0.0434 0.0678 0.0681 0.0499
#              0.0434 1.0000 0.1858 0.7079 0.5195
#              0.0678 0.1858 1.0000 0.1429 0.1520
#              0.0681 0.7079 0.1429 1.0000 0.5381
#              0.0499 0.5195 0.1520 0.5381 1.0000

cor(num_class, !group2_na)
# -0.120 - seems to be a decent feature, no measurements -> lower prob of a failure

#########################################################################################################
# Looking at Group 3: "ct_000", "cu_000", "cv_000", "cx_000", "cy_000", "cz_000", "da_000", "db_000", "dc_000"

cases <- (!is.na(ndata$ct_000))&(ndata$ct_000<1e4)
hist(ndata$ct_000[cases], prob=TRUE, las=1, breaks=100, xlab="Values of ct_000 feature", main="Distribution of ct_000 values", ylab="Probability density")
# Checked all the Group 3 - similar picture

cases <- !is.na(ndata$ct_000)
logvar <- log(ndata$ct_000 + 1)
hist(logvar, prob=TRUE, las=1, breaks=100, xlab="Values of log-transformed ct_000", ylab="Probability density", main="Distribution of log ct_000")
lines(density(logvar[cases]), col=2) # 2-bell

logvar <- log(ndata$cu_000 + 1)
hist(logvar, prob=TRUE, las=1, breaks=100)
lines(density(logvar[cases]), col=2) # 2-bell

logvar <- log(ndata$cv_000 + 1)
hist(logvar, prob=TRUE, las=1, breaks=100)
lines(density(logvar[cases]), col=2)  # Wow! Nice 3-bell shape

logvar <- log(ndata$cx_000 + 1)
hist(logvar, prob=TRUE, las=1, breaks=100)
lines(density(logvar[cases]), col=2) # 2-bell

logvar <- log(ndata$cy_000 + 1)
hist(logvar, prob=TRUE, las=1, breaks=100) # 85% zeros
cases <- !is.na(logvar)&(logvar>0)
hist(logvar[cases], prob=TRUE, las=1, breaks=100)
lines(density(logvar[cases]), col=2) # Though there is a 1-bell shape
cases <- !is.na(ndata$cy_000)

logvar <- log(ndata$cz_000 + 1)
hist(logvar, prob=TRUE, las=1, breaks=100)
lines(density(logvar[cases]), col=2) # 1-bell

logvar <- log(ndata$da_000 + 1)
hist(logvar, prob=TRUE, las=1, breaks=100) # 98.8% zeros
cases <- !is.na(logvar)&(logvar>0)
hist(logvar[cases], prob=TRUE, las=1, breaks=100)
lines(density(logvar[cases]), col=2) # Though there is a 1-bell shape
cases <- !is.na(ndata$da_000)

logvar <- log(ndata$db_000 + 1)
hist(logvar, prob=TRUE, las=1, breaks=100)
lines(density(logvar[cases]), col=2) # 1-bell

logvar <- log(ndata$dc_000 + 1)
hist(logvar, prob=TRUE, las=1, breaks=100)
lines(density(logvar[cases]), col=2) # Wow! Nice 3-bell shape

#########################################################################################################
# In Group 3: "ct_000", "cu_000", "cv_000", "cx_000", "cy_000", "cz_000", "da_000", "db_000", "dc_000"
# Apply log(x+1) to all non-missing values
# The variables have a 1, 2, and 3-bell shapes and a peak at zero
# It seems reasonable to leave transformed features even with over 20% NAs, 
# Impute instead of NAs zero with probability of zero and median-of-non-zero with probability of non-zero
# Also, "dc_000" and "cv_000" are highly correlated for complete cases - drop "cv_000"
#########################################################################################################

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
#	print(colnames(ndata)[label]); print(nzeros); print(total); print(med)
}

pairs_data <- cbind(num_class, ndata$ct_000, ndata$cu_000, ndata$cx_000, ndata$cy_000, ndata$cz_000, ndata$da_000, ndata$db_000, ndata$dc_000)
options(digits=3)
cor(pairs_data)

cor(num_class, !group3_na)
# [1] -0.09336662 - added this dummy

#########################################################################################################
# Looking at the rest of problematic variables

suspect_labels <- c("bk_000", "bl_000", "bm_000", "bn_000", "bo_000", "bp_000", "bq_000", "br_000", "cl_000", "cm_000", "ec_00", "ed_000")
m <- length(suspect_labels)
names <- rep("",m)
maxs <- rep(0,m)
zeros <- rep(0,m)
cors <- rep(0,m)
log_cors <- rep(0,m)
not_na <- rep(0,m)

for(j in 1:m){
	i <- which(colnames(ndata)==suspect_labels[j])
	names[j] <- colnames(ndata)[i]
	cases <- !is.na(ndata[,i])
	not_na[j] <- sum(cases)
	zeros[j] <- sum(ndata[cases,i]==0)
	maxs[j] <- max(ndata[cases,i])
	cors[j] <- cor(num_class[cases], ndata[cases,i])
	log_cors[j] <- cor(num_class[cases], log(ndata[cases,i]+1))
}

rbind(names, not_na, zeros, maxs, cors, log_cors, options(digits=3))
# names bk_000 bl_000   bm_000  bn_000  bo_000  bp_000  bq_000  br_000  cl_000  cm_000   ec_00  ed_000
# zeros  502    1016    1214     1422    1503    1612    1734    1813    36787   24215   1276   1418
# z/t    0.013  0.031   0.059    0.088   0.109   0.131   0.153   0.168   0.729   0.483   0.025  0.028	# proportion of zeros
# maxs   1.3e6  1.3e6   1.3e6    1.3e6   1.3e6   1.3e6   1.3e6   1.3e6   1.3e5   7.3e4   1.1e5  8.3e4	# need log-transformation
# cors   0.066  0.025  -0.021   -0.057  -0.086  -0.110  -0.131  -0.143   0.184   0.373   0.372  0.395	# correlation with pos class
# cors_l 0.061  0.049   0.058    0.069   0.077   0.084   0.093   0.101   0.260   0.131   0.098  0.113	# cor of log-transformation with pos class
# All features seem to contribute. Max-s are high, so for stat methods to work properly we will need transformation

#########################################################################################################
# In "Other suspects": "bk_000", "bl_000", "bm_000", "bn_000", "bo_000", "bp_000", "bq_000", "br_000", "cl_000", "cm_000", "ec_00", "ed_000"
# Apply log(x+1) to all non-missing values
# Impute instead of NAs zero with probability of zero and median-of-non-zero with probability of non-zero
#########################################################################################################

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
#	print(colnames(ndata)[label]); print(nzeros); print(total); print(med)
}

pairs_data <- cbind(num_class, ndata$bk_000, ndata$bl_000, ndata$bm_000, ndata$bn_000, ndata$bo_000, ndata$bp_000, ndata$bq_000, ndata$br_000, ndata$cl_000, ndata$cm_000, ndata$ec_00, ndata$ed_000)
options(digits=3)
cor(pairs_data)
# All features seem to contribute and no multicollinearity issues can be seen from pairwise correlations

#########################################################################################################
# Histogram features

attach(ndata)
ag <- cbind(ag_000, ag_001, ag_002, ag_003, ag_004, ag_005, ag_006, ag_007, ag_008, ag_009)
ay <- cbind(ay_000, ay_001, ay_002, ay_003, ay_004, ay_005, ay_006, ay_007, ay_008, ay_009)
az <- cbind(az_000, az_001, az_002, az_003, az_004, az_005, az_006, az_007, az_008, az_009)
ba <- cbind(ba_000, ba_001, ba_002, ba_003, ba_004, ba_005, ba_006, ba_007, ba_008, ba_009)
cn <- cbind(cn_000, cn_001, cn_002, cn_003, cn_004, cn_005, cn_006, cn_007, cn_008, cn_009)
cs <- cbind(cs_000, cs_001, cs_002, cs_003, cs_004, cs_005, cs_006, cs_007, cs_008, cs_009)
ee <- cbind(ee_000, ee_001, ee_002, ee_003, ee_004, ee_005, ee_006, ee_007, ee_008, ee_009)
detach(ndata)
summary(ag) 
# Look at each histogram and recognize the same pattern of values distribution
# So we log transform these features
ag <- log(ag+1)
ay <- log(ay+1)
az <- log(az+1)
ba <- log(ba+1)
cn <- log(cn+1)
cs <- log(cs+1)
ee <- log(ee+1)

# Check the pos class rate among NAs - it is much lower than avg=0.01667
sum(num_class[is.na(ag[,1])])/sum(is.na(ag[,1])) # 0.0060 pos class among NAs
sum(num_class[is.na(ay[,1])])/sum(is.na(ay[,1])) # 0.0075 pos class among NAs
sum(num_class[is.na(az[,1])])/sum(is.na(az[,1])) # 0.0075 pos class among NAs
sum(num_class[is.na(ba[,1])])/sum(is.na(ba[,1])) # 0.0058 pos class among NAs
sum(num_class[is.na(cn[,1])])/sum(is.na(cn[,1])) # 0.0058 pos class among NAs
sum(num_class[is.na(cs[,1])])/sum(is.na(cs[,1])) # 0.0060 pos class among NAs
sum(num_class[is.na(ee[,1])])/sum(is.na(ee[,1])) # 0.0075 pos class among NAs

# We will capture this difference by a new dummy feature on histogram NAs: 
hist_na <- (is.na(ag[,1]))|(is.na(ay[,1]))|(is.na(az[,1]))|(is.na(ba[,1]))|(is.na(cn[,1]))|(is.na(cs[,1]))|(is.na(ee[,1]))

# Look at the pattern of data
ag[300:350,]
# See that the data goes in kind of waves, so average'under the curve' or argmax can be of interest

# New variables for the histogram features - average across histogram 
ag_m <- apply(ag,1,mean); ag_m[is.na(ag_m)] <- median(ag_m, na.rm=T)
ay_m <- apply(ay,1,mean); ay_m[is.na(ay_m)] <- median(ay_m, na.rm=T)
az_m <- apply(az,1,mean); az_m[is.na(az_m)] <- median(az_m, na.rm=T)
ba_m <- apply(ba,1,mean); ba_m[is.na(ba_m)] <- median(ba_m, na.rm=T)
cn_m <- apply(cn,1,mean); cn_m[is.na(cn_m)] <- median(cn_m, na.rm=T)
cs_m <- apply(cs,1,mean); cs_m[is.na(cs_m)] <- median(cs_m, na.rm=T)
ee_m <- apply(ee,1,mean); ee_m[is.na(ee_m)] <- median(ee_m, na.rm=T)

pairs_data <- cbind(num_class, ag_m, ay_m, az_m, ba_m, cn_m, cs_m, ee_m)
cor(pairs_data)
# Correlations of the "mean/sum" for each histogram with the class is good for every block
# We may try to leave only this variable for each histogram, or use original features when conducting analysis

# As some of the variables have a distinct peak and pos rate depends on its position, we introduce argmax 
ag_f <- rep(0,nrow(ndata))
ay_f <- rep(0,nrow(ndata))
az_f <- rep(0,nrow(ndata))
ba_f <- rep(0,nrow(ndata))
cn_f <- rep(0,nrow(ndata))
cs_f <- rep(0,nrow(ndata))
ee_f <- rep(0,nrow(ndata))

for(i in 1:nrow(ndata)){
	if (hist_na[i]) {
		ag_f[i] <- 0; ay_f[i] <- 0; az_f[i] <- 0; ba_f[i] <- 0; cn_f[i] <- 0; cs[i] <- 0; ee_f[i] <- 0
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

# Check how argmax is associated with pos class rate - as a convex function for some histograms or not that clear for others

a <- rep(0,10); b<- rep(0,10)
for(i in 0:9){
	a[i+1] <- sum(ag_f==i, na.rm=T)
	b[i+1] <- sum(num_class[ag_f==i], na.rm=T)/a[i+1]
}
cbind(num_of_maxes=a,failures_when_max_in_this_bin=b)

#########################################################################################################
# In "histograms"
# Apply log(x+1) to all non-missing values
# Add mean-in-a-histogram feature and argmax-in-a-histogram feature - 7+7=14 features
# Impute instead of NAs a median for this cell in a histogram
# Get a distribution of medians in a missing histogram - not a median distribution
#########################################################################################################

ndata <- cbind(ndata, hist_na=ifelse(hist_na==TRUE, 1, 0))
ndata <- cbind(ndata, ag_m, ay_m, az_m, ba_m, cn_m, cs_m, ee_m, ag_f, ay_f, az_f, ba_f, cn_f, cs_f, ee_f)

# Imputing medians for NAs
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

# Return transformed histogram values to the dataset
j <- which(colnames(ndata)=="ag_000"); ndata[,j:(j+9)] <- ag
j <- which(colnames(ndata)=="ay_000"); ndata[,j:(j+9)] <- ay
j <- which(colnames(ndata)=="az_000"); ndata[,j:(j+9)] <- az
j <- which(colnames(ndata)=="ba_000"); ndata[,j:(j+9)] <- ba
j <- which(colnames(ndata)=="cn_000"); ndata[,j:(j+9)] <- cn
j <- which(colnames(ndata)=="cs_000"); ndata[,j:(j+9)] <- cs
j <- which(colnames(ndata)=="ee_000"); ndata[,j:(j+9)] <- ee

#########################################################################################################
# Treating non-suspects and non-histogram features

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

others_flag[1] <- FALSE # that's the class variable, not a feature

sum(others_flag)
other_labels <- colnames(ndata)[others_flag]

# Labels of the features to deal with
other_labels

# Look at the zeros rate and corelation with pos class for features
a <- c("",0,0,0,0,0)
b <- c("",0,0,0,0,0)
for(i in 1:length(others_flag)){
	if(others_flag[i]==TRUE){
		a[1] <- colnames(ndata)[i]
		a[2] <- sum(ndata[,i]==0, na.rm=T)		# zero values
		a[3] <- sum(ndata[,i]>0, na.rm=T)		# non-zero values
		a[4] <- sum(is.na(ndata[,i]))			# NA values
		a[5] <- cor(ndata[!is.na(ndata[,i]),i], num_class[!is.na(ndata[,i])])	# complete cases correlation with pos class
		a[6] <- cor(is.na(ndata[,i]), num_class)	# correlation of missingness with pos class
		b <- rbind(b,a)		
	}

}
b

# label     zeros   nonzeros  NAs   cor     cor-is.na  comments
#  aa_000     393     59607   0     0.536    NA      +
#  ac_000     8752    47913   3335 -0.036    0.2309  add dummy  ac_na 
#  ae_000     55543   1957    2500  0.008    0.195   add dummy  group2500 
#  af_000     55476   2024    2500  0.022    0.195   group2500
#  ah_000     133     59222   645   0.524    0.0381  +
#  ai_000     53588   5783    629   0.121    0.040   +
#  aj_000     46919   12452   629   0.024    0.040   +
#  ak_000     55227   373     4400  0.018    0.229   +
#  al_000     37473   21885   642   0.373    0.0408  +
#  am_0       37295   22076   629   0.380    0.040   +
#  an_000     129     59229   642   0.524    0.040   +
#  ao_000     140     59271   589   0.519    0.039   +
#  ap_000     113     59245   642   0.515    0.040   +
#  aq_000     230     59181   589   0.531    0.039   +
#  ar_000     54928   2349    2723  0.185    0.207   add dummy  group2723 
#  as_000     59350   21      629   0.044    0.040   drop  as_000 
#  at_000     53395   5976    629   0.139    0.040   +
#  au_000     59310   61      629   0.093    0.040   drop  au_000 
#  av_000     10008   47492   2500  0.156    0.195   group2500
#  ax_000     10191   47308   2501  0.142    0.194   group2500
#  bb_000     107     59248   645   0.542    0.038   +
#  bc_000     19296   37979   2725  0.236    0.207   group2723
#  bd_000     4030    53243   2727  0.230    0.207   group2723
#  be_000     6951    50546   2503  0.151    0.194   group2500
#  bf_000     25294   32206   2500  0.213    0.195   group2500
#  bg_000     133     59225   642   0.522    0.040   +
#  bh_000     129     59229   642   0.497    0.040   +
#  bi_000     117     59294   589   0.429    0.039   +
#  bj_000     137     59274   589   0.525    0.039   +
#  bs_000     120     59154   726   0.187    0.039   +
#  bt_000     131     59702   167   0.537    0.020   +
#  bu_000     107     59202   691   0.541    0.037   +
#  bv_000     107     59202   691   0.541    0.037   drop  bv_000  almost copy of bu_000
#  bx_000     0       56743   3257  0.512   -0.015   +
#  by_000     1224    58303   473   0.500   -0.010   +
#  bz_000     13841  43436   2723   0.201    0.207   group2723
#  ca_000     109    55535   4356   0.052    0.229   +
#  cb_000     109    59165   726    0.064    0.039   +
#  cc_000     878    55867   3255   0.519    0.015   +
#  cd_000     0      59324   676    NA       0.039   drop  cd_000 
#  ce_000     11845  45653   2502   0.243    0.194   group2500
#  ci_000     223    59439   338    0.553    0.007   +
#  cj_000     47122  12540   338    0.308    0.007   +
#  ck_000     194    59468   338    0.466    0.007   +
#  cp_000     9537   47739   2724   0.066    0.207   group2723
#  cq_000     107    59202   691    0.541    0.037   +
#  dd_000     866    56631   2503   0.350    0.194   group2500
#  de_000     860    56416   2724   0.247    0.207   group2723
#  df_000     54930  1062    4008   0.128    0.238   add dummy group4008
#  dg_000     54424  1568    4008   0.192    0.238   group4008
#  dh_000     49008  6984    4008   0.044    0.238   group4008
#  di_000     45647  10347   4006   0.242    0.238   group4008
#  dj_000     55796  197     4007  -0.000    0.238   drop  dj_000 
#  dk_000     55675  318     4007   0.019    0.238   drop  dk_000 
#  dl_000     55769  223     4008  -0.002    0.238   drop  dl_000 
#  dm_000     55734  257     4009   0.006    0.238   drop  dm_000 
#  dn_000     107    59202   691    0.493    0.037   +
#  do_000     13287  43989   2724   0.355    0.207   group2723
#  dp_000     12750  44524   2726   0.333    0.207   group2723
#  dq_000     44404  12870   2726   0.110    0.207   group2723
#  dr_000     44267  13007   2726   0.198    0.207   group2723
#  ds_000     2662   54611   2727   0.417    0.207   group2723
#  dt_000     2551   54722   2727   0.419    0.207   group2723
#  du_000     3280   53994   2726   0.208    0.207   group2723
#  dv_000     3278   53996   2726   0.249    0.207   group2723
#  dx_000     39558  17719   2723   0.187    0.207   group2723
#  dy_000     36269  21007   2724   0.137    0.207   group2723
#  dz_000     56984  293     2723   0.040    0.207   drop  dz_000 
#  ea_000     56535  742     2723   0.010    0.207   drop  ea_000 
#  eb_000     21449  34544   4007   0.234    0.238   group4008
#  ef_000     57021  255     2724   0.018    0.207   drop  ef_000 
#  eg_000     56794  483     2723   0.017    0.207   drop  eg_000 

# Another cleaning will be applied for highly correlated features

cortable <- cor(ndata[,others_flag],use="pairwise.complete.obs")
dim(cortable)
# 89 89

# Pick out the highly correlated pairs from cor-table
a <- c("","",0)
b <- c("","",0)
for (i in 1:sum(others_flag)){
	for(j in 1:sum(others_flag)){
 		if(!is.na(cortable[j,i])&(cortable[j,i]>0.95)&(cortable[j,i]<1)){
 			a[1] <- rownames(cortable)[j]
 			a[2] <- colnames(cortable)[i]
 			a[3] <- cortable[i,j]
 			b <- rbind(b, a)
 		}
 	}
 }
b

#	x	y		cor		action
# "al_000"    "am_0"      "0.993894781996434"	drop "al_000"
# "ah_000"    "an_000"    "0.988374544930902"	drop "ah_000"
# "ah_000"    "ao_000"    "0.974144741658944"	+
# "an_000"    "ao_000"    "0.992996809337373"	drop "ao_000"
# "ah_000"    "bb_000"    "0.983378730736376"	+
# "an_000"    "bb_000"    "0.988159535695112"	drop "bb_000"
# "ao_000"    "bb_000"    "0.977493452522328"	+
# "ah_000"    "bg_000"    "0.999999999999824"	+
# "an_000"    "bg_000"    "0.988341881871882"	drop "bg_000"
# "ao_000"    "bg_000"    "0.974085225993898"	+
# "bb_000"    "bg_000"    "0.983363037265188"	+
# "ah_000"    "bh_000"    "0.953197042916723"	+
# "aq_000"    "bh_000"    "0.963333566487338"	drop "aq_000"
# "bg_000"    "bh_000"    "0.953211766241036"	+
# "aq_000"    "bj_000"    "0.952326865685983"	+
# "aa_000"    "bt_000"    "0.999999999992162"	drop "bt_000"
# "ah_000"    "bu_000"    "0.983374732284039"	+
# "an_000"    "bu_000"    "0.98814708719439" 	+
# "ao_000"    "bu_000"    "0.977487824125287"	+
# "bb_000"    "bu_000"    "0.999999999989625"	+
# "bg_000"    "bu_000"    "0.983351977299334"	+
# "ah_000"    "bv_000"    "0.983374746376541"	+
# "an_000"    "bv_000"    "0.988147096353803"	dropped "bv_000" earlier
# "ao_000"    "bv_000"    "0.977487830039383"	+
# "bb_000"    "bv_000"    "0.99999999999992" 	+
# "bg_000"    "bv_000"    "0.983351991406748"	+
# "bu_000"    "bv_000"    "0.999999999989653"	+
# "bx_000"    "cc_000"    "0.996185646979512"	drop "bx_000"
# "aa_000"    "ci_000"    "0.9653458590664"  	drop "ci_000"
# "bt_000"    "ci_000"    "0.966735519120933"	+
# "ah_000"    "cq_000"    "0.983374738740146"	+
# "an_000"    "cq_000"    "0.988147091988418"	drop "cq_000"
# "ao_000"    "cq_000"    "0.977487829565958"	+
# "bb_000"    "cq_000"    "0.999999999998711"	+
# "bg_000"    "cq_000"    "0.983351983762981"	+
# "bu_000"    "cq_000"    "0.999999999990911"	+
# "bv_000"    "cq_000"    "0.999999999998733"	+
# "aq_000"    "dn_000"    "0.969294781250831"	+
# "bh_000"    "dn_000"    "0.974499681423253"	drop "dn_000"
# "ds_000"    "dt_000"    "0.969448180059223"	drop "ds_000"

#########################################################################################################
# In "non-suspects"
# Create dummy on "ac_000"-NA, group with around 2500 NAs, group with around 2723 NAs, and group of 4800 NAs
# Those features "are missing together" and the fact of their missingness explains pos class well
# Drop the list of variables with too many zeros or too correlated 
# Log-transform all the original features
# Impute by MICE
#########################################################################################################
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

#########################################################################################################

library(mice)

colnames(ndata)
# 21 additional features at the end of the dataset
# log-transform original variables only

idata <- ndata[,others_flag] 
idata[, 1:((ncol(idata)-20))] <- log(idata[,1:(ncol(idata)-20)]+1)
summary(idata)

tempdata <- mice(idata, m=1, maxit=3, meth='pmm', seed=1)
idata <- complete(tempdata,1) # imputed 1 dataset, though more can be done

ndata[,others_flag] <- idata
summary(ndata)

#########################################################################################################
# Chose to drop due to multicollinearity more features:

cortable <- cor(ndata[,-1])
write.csv(cortable, "correlations.csv")

drop_labels <- c("af_000", "an_000", "ap_000", "bh_000", "bj_000", "cc_000", "ct_000", "ac_na")
for(j in 1:length(drop_labels)){
	drop <- which(colnames(ndata)==drop_labels[j])
	ndata <- ndata[,-drop]
}
dim(ndata)

##########################################################################################################################
write.csv(ndata, file="use_me_train_data.csv")




