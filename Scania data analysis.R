#############################################################################################

library(class)
library(caretEnsemble)
library(caret)

# Reading pre-processed data into R

data <- read.csv(file="use_me_train_data.csv", header=TRUE)
dim(data); colnames(data)

tdata <- read.csv(file="use_me_test_data.csv", header=TRUE)
dim(tdata); 

num_class <- ifelse(data$class=="pos",1,0)
test_class <- ifelse(tdata$class=="pos",1,0)

#############################################################################################

# Making principal components features for the training dataset

pca_data <- data[,-c(1,2)]
pr.out <- prcomp(pca_data, scale=TRUE)
pr.var <- pr.out$sdev^2
pve <- pr.var/sum(pr.var)

plot(cumsum(pve), ylab="Proportion of variance explained", main="Variance explained by principal components", xlab="Number of principal components used")
which(cumsum(pve)>0.959)[1]

# Take the first 100 features, as they explain almost 99% of variation

#############################################################################################
# Fitting glm logistic model on 'PCA' training dataset
pca_data <- data.frame(cbind(num_class, pr.out$x[,1:100]))
glm.fit <- glm(num_class~., data=pca_data, family="binomial")
fitted_class <- predict(glm.fit, newdata = pca_data[,-1], type="response")

# Find the min cost probability threshold, based on the output of the glm model

# As an improvement we could have done it by cross-validation, though in our dataset
# pos class rate is ~1.6%, so we should have done splitting of data carefully

threshold <- seq(from=0.001, to=0.2, by=0.001)
total_cost <- rep(0, length(threshold))

for(i in 1:length(threshold)){
  fitted_pos <- (fitted_class>=threshold[i])
  total_cost[i] <- 10*sum((fitted_pos==1)&(num_class==0))+500*sum((fitted_pos==0)&(num_class==1))
}

plot(cbind(threshold,total_cost))
i <- which.min(total_cost)
pca_threshold <- threshold[i] # 0.022, use for prediction on test data

#############################################################################################

# Testing glm logistic model on 'PCA' testing dataset
pca_data <- predict(pr.out, newdata=tdata[, -c(1,2)])[,1:100]
pca_data <- data.frame(cbind(test_class, pca_data))
pred_class <- predict(glm.fit, newdata=pca_data, type="response")


# Find the predicted class with the estimated threshold on predicted 'probability'
pred_pos <- (pred_class>=pca_threshold)
total_cost <- 10*sum((pred_pos==1)&(test_class==0))+500*sum((pred_pos==0)&(test_class==1))
total_cost
table(pred_pos, test_class)

#  Total cost on PCA dataset = 15,470
#         test_class
# pred_pos     0     1
#    FALSE 14978    18
#    TRUE    647   357

#############################################################################################
# Fitting LDA and QDA models on 'PCA' training dataset

library(MASS)

pca_data <- data[,-c(1,2)]
pr.out <- prcomp(pca_data, scale=TRUE)
pca_train <- data.frame(cbind(num_class, pr.out$x[,1:100]))

lda_fit <- lda(num_class~., data=pca_train)
qda_fit <- qda(num_class~., data=pca_train)

lda_fitted_class <- predict(lda_fit, newdata=pca_train[,-1])$posterior[,2]
qda_fitted_predict <- predict(qda_fit, newdata=pca_train[,-1])$posterior[,2]

# Find the min cost probability threshold, based on the output of the glm model

threshold <- seq(from=0, to=5e-2, by=1e-6)

total_cost <- matrix(rep(0, 2*length(threshold)), ncol=2)
for(i in 1:length(threshold)){
  lda_fitted_pos <- (lda_fitted_class>=threshold[i])
  qda_fitted_pos <- (qda_fitted_class>=threshold[i])
  total_cost[i,1] <- 10*sum((lda_fitted_pos==1)&(num_class==0))+500*sum((lda_fitted_pos==0)&(num_class==1))
  total_cost[i,2] <- 10*sum((qda_fitted_pos==1)&(num_class==0))+500*sum((qda_fitted_pos==0)&(num_class==1))
}

plot(cbind(threshold,total_cost[,1]))

i <- which.min(total_cost[,1])
lda_threshold <- threshold[i] # 1e-6 use for prediction on test data
plot(cbind(threshold,total_cost[,2]))
i <- which.min(total_cost[,2])
qda_threshold <- threshold[i] # 0.0102 use for prediction on test data

#############################################################################################

# Testing LDA and QDA models on 'PCA' testing dataset

pca_data <- predict(pr.out, newdata=tdata[, -c(1,2)])[,1:100]
pca_test <- data.frame(cbind(test_class, pca_data))

lda_pred_class <- predict(lda_fit, newdata=pca_test)$posterior[,2]
qda_pred_class <- predict(qda_fit, newdata=pca_test)$posterior[,2]

lda_pos <- (lda_pred_class>=lda_threshold)
total_cost <- 10*sum((lda_pos==1)&(test_class==0))+500*sum((lda_pos==0)&(test_class==1))

table(lda_pos, test_class)
#  Total cost for LDA on PCA dataset = 25,120
# lda_pos     0     1
#   FALSE 14913    36
#   TRUE   712    339

qda_pos <- (qda_pred_class>=qda_threshold)
total_cost <- 10*sum((qda_pos==1)&(test_class==0))+500*sum((qda_pos==0)&(test_class==1))

table(qda_pos, test_class)
#  Total cost for LDA on PCA dataset = 20,350
#       test_class
# qda_pos     0     1
#   FALSE 14617    19
#   TRUE   1008   356

#############################################################################################
# KNN on PCA data

library(class)

train.X <- pr.out$x[,1:100]
train.Y <- as.factor(num_class)
set.seed(1)
subX <- sample(1:nrow(train.X), 30000) 
set.seed(2)
subY <- sample(1:nrow(train.X),10000)
test.X <- predict(pr.out, newdata=tdata[,-c(1,2)])[,1:100]

knn.fit <- knn(train.X[subX,], train.X[subY,], train.Y[subX], k=3, prob=T)
knn.pred <- attributes(knn.fit)$prob

threshold <- seq(from=0, to=1, by=0.01)
total_cost <- rep(0, length(threshold))
for (i in 1:length(threshold)){
  knn.pos <- (knn.pred>=threshold[i])
  total_cost[i] <- sum(10*((knn.pos==1)&(train.Y[subY]==0))+500*((knn.pos==0)&(train.Y[subY]==1)))
}
plot(threshold, total_cost)
knn_threshold <- 0.5 # unfortunately, not quite informative, maybe, because it is a portion of votes, not a probability measure

knn.fit <- knn(train.X, test.X, train.Y, k=3, prob=T)
total_cost <- sum(10*((knn.fit==1)&(test_class==0))+500*((knn.fit==0)&(test_class==1)))
total_cost
table(knn.fit, test_class)
# For PCA training dataset total_cost = 80,830
#       test_class
# knn.fit     0     1
#       0 15592   161
#       1    33   214

#############################################################################################

# Fitting the SVM model with linear kernel on PCA data

library(e1071)

pca_data <- data[,-c(1,2)]
pr.out <- prcomp(pca_data, scale=TRUE)
train_data <- data.frame(cbind(num_class,pr.out$x[,1:50]))

set.seed(1)
tunes <- sample(1:nrow(train_data),10000)
set.seed(2)
tune.out <- tune(svm, num_class~., data=train_data[tunes,], kernel="linear", ranges=list(cost=c(0.1,1,10)))

svm_pca <- tune.out$best.model
summary(svm_pca)
# cost=1, gamma=0.02, eps=0.1

svm_pca <- svm(num_class~., data=train_data, kernel ="linear", cost=1, gamma=0.02, epsilon=0.1, scale = T)
summary(svm_pca)

svm_fitted_pred <- predict(svm_pca, newdata=train_data)

threshold <- seq(from=0.001, to=0.1, by=0.001)
total_cost <- rep(0, length(threshold))
for(i in 1:length(threshold)){
  svm_fitted_pos <- (svm_fitted_pred>=threshold[i])
  total_cost[i] <- 10*sum((svm_fitted_pos==1)&(num_class==0))+500*sum((svm_fitted_pos==0)&(num_class==1))
}

plot(threshold,total_cost)
svm_threshold <- threshold[which.min(total_cost)] 
# optimal threshold = 0.005

#############################################################################################
# SVM model testing on PCA of test data

pca_data <- predict(pr.out, newdata=tdata[, -c(1,2)])[,1:50]
test_data <- data.frame(cbind(test_class, pca_data))

svm_pred <- predict(svm_pca, newdata=test_data[,-1])
svm_pos <- (svm_pred >= svm_threshold)
total_cost <- 10*sum((svm_pos==1)&(test_class==0))+500*sum((svm_pos==0)&(test_class==1))
total_cost
table(svm_pos, test_class)

# For PCA training dataset total_cost = 31,670
#       test_class
# svm_pos     0     1
#   FALSE   14358   38
#   TRUE    1267   337

#############################################################################################

# Over/Under/No sampling for highly imbalanced dataset

# clear out the histogram features 
hist_labels <- c("ag_000", "ag_001", "ag_002", "ag_003", "ag_004", "ag_005", "ag_006", "ag_007", "ag_008", "ag_009", "ay_000", "ay_001", "ay_002", "ay_003", "ay_004", "ay_005", "ay_006", "ay_007", "ay_008", "ay_009", "az_000", "az_001", "az_002", "az_003", "az_004", "az_005", "az_006", "az_007", "az_008", "az_009", "ba_000", "ba_001", "ba_002", "ba_003", "ba_004", "ba_005", "ba_006", "ba_007", "ba_008", "ba_009", "cn_000", "cn_001", "cn_002", "cn_003", "cn_004", "cn_005", "cn_006", "cn_007", "cn_008", "cn_009", "cs_000", "cs_001", "cs_002", "cs_003", "cs_004", "cs_005", "cs_006", "cs_007", "cs_008", "cs_009", "ee_000", "ee_001", "ee_002", "ee_003", "ee_004", "ee_005", "ee_006", "ee_007", "ee_008", "ee_009")
features_flag <- rep(TRUE, ncol(data))

for(j in 1:length(hist_labels)){
  i <- which(colnames(data)==hist_labels[j])
  features_flag[i] <- FALSE
}
features_flag[1:2] <- FALSE

# Features flag picks relevant variables, for histograms - mean and argmax

sampling_data <- cbind(num_class,data[,features_flag])

library(ROSE)
# oversampling of the minority class
ovdata <- ovun.sample(num_class~., data=sampling_data, method="over", p=0.5, seed=1)$data
table(ovdata$num_class)

# undersampling of the majority class
undata <- ovun.sample(num_class~., data=sampling_data, method="under", p=0.5, seed=1)$data
table(undata$num_class)

###############################################################################################

# Fitting glm logistic model on over-under-original training dataset

# No histogram features used, only mean/argmax synthetic ones

glm_over <- glm(num_class~., data=ovdata, family="binomial")
glm_under <- glm(num_class~., data=undata, family="binomial")
glm_original <- glm(num_class~., data=sampling_data, family="binomial")

# Find the min cost probability threshold, based on the output of the glm model

fitted_over <- predict(glm_over, newdata=ovdata[,-1], type="response")
fitted_under <- predict(glm_under, newdata=undata[,-1], type="response")
fitted_original <- predict(glm_original, newdata=sampling_data[,-1], type="response")

threshold <- seq(from=0.01, to=0.4, by=0.01)
total_cost <- matrix(rep(0, 3*length(threshold)),ncol=3)

for(i in 1:length(threshold)){
  over_fitted_pos <- (fitted_over>=threshold[i])
  under_fitted_pos <- (fitted_under>=threshold[i])
  original_fitted_pos <- (fitted_original>=threshold[i])
  total_cost[i,1] <- 10*sum((over_fitted_pos==1)&(ovdata[,1]==0))+500*sum((over_fitted_pos==0)&(ovdata[,1]==1))
  total_cost[i,2] <- 10*sum((under_fitted_pos==1)&(undata[,1]==0))+500*sum((under_fitted_pos==0)&(undata[,1]==1))
  total_cost[i,3] <- 10*sum((original_fitted_pos==1)&(sampling_data[,1]==0))+500*sum((original_fitted_pos==0)&(sampling_data[,1]==1))
}

plot(threshold,total_cost[,1]) # over-sampled
plot(threshold,total_cost[,2]) # under-sampled

plot(threshold,total_cost[,3])
original_threshold <- threshold[which.min(total_cost[,3])] 

over_threshold <- 0.5 
under_threshold <- 0.5 

#############################################################################################

# Testing glm logistic model on the testing dataset

over_pred <- predict(glm_over, newdata=tdata[, features_flag], type="response")
under_pred <- predict(glm_under, newdata=tdata[, features_flag], type="response")
original_pred <- predict(glm_original, newdata=tdata[, features_flag], type="response")

over_pos <- (over_pred >= over_threshold)
total_cost <- 10*sum((over_pos==1)&(test_class==0))+500*sum((over_pos==0)&(test_class==1))

table(over_pos, test_class)
# For oversampled training dataset total_cost = 14,580

#        test_class
#over_pos     0     1
#   FALSE 14917    15
#   TRUE   708    360

under_pos <- (under_pred >= under_threshold)
total_cost <- 10*sum((under_pos==1)&(test_class==0))+500*sum((under_pos==0)&(test_class==1))

table(under_pos, test_class)
# For undersampled training dataset total_cost = 15,590
#         test_class
#under_pos     0     1
#    FALSE 14866    16
#    TRUE   759    359

original_pos <- (original_pred >= original_threshold)
total_cost <- 10*sum((original_pos==1)&(test_class==0))+500*sum((original_pos==0)&(test_class==1))

table(original_pos, test_class)

# For original training dataset total_cost = 14,090
#            test_class
#original_pos     0     1
#       FALSE 14916    14
#       TRUE    709   361

###############################################################################################

# Fitting LDA and QDA models on over-under-original training dataset
# No histogram features used, only mean/argmax synthetic ones

lda_over <- lda(num_class~., data=ovdata)
lda_under <- lda(num_class~., data=undata)
lda_original <- lda(num_class~., data=sampling_data)

qda_over <- qda(num_class~., data=ovdata)
qda_under <- qda(num_class~., data=undata)
qda_original <- qda(num_class~., data=sampling_data)

# Find the min cost probability threshold, based on the output of the glm model

lda_fitted_original <- predict(lda_original, newdata=sampling_data[,-1])$posterior[,2]
qda_fitted_original <- predict(qda_original, newdata=sampling_data[,-1])$posterior[,2]

threshold <- seq(from=0, to=0.045, by=1e-3)
total_cost <- matrix(rep(0, 2*length(threshold)),ncol=2) 
for(i in 1:length(threshold)){
  
  lda_original_fitted_pos <- (lda_fitted_original>=threshold[i])
  total_cost[i,1] <- 10*sum((lda_original_fitted_pos==1)&(sampling_data[,1]==0))+500*sum((lda_original_fitted_pos==0)&(sampling_data[,1]==1))
  qda_original_fitted_pos <- (qda_fitted_original>=threshold[i])
  total_cost[i,2] <- 10*sum((qda_original_fitted_pos==1)&(sampling_data[,1]==0))+500*sum((qda_original_fitted_pos==0)&(sampling_data[,1]==1))
}

lda_original_threshold <- threshold[which.min(total_cost[,1])] # 0.001 yields a min for the original dataset
qda_original_threshold <- threshold[which.min(total_cost[,2])] # 0.042 yields a min for the original dataset

lda_over_threshold <- 0.5  # 0.010 would yield a min for oversampled dataset, and it won't perform on unbalanced test data (total_cost = 29,990)
lda_under_threshold <- 0.5  # 0.009 would yield a min for undersampled dataset, and it won't perform on unbalanced test data (total_cost = 31,910)
qda_over_threshold <- 0.5  
qda_under_threshold <- 0.5 

#############################################################################################

# Testing LDA model on the testing dataset

lda_over_pred <- predict(lda_over, newdata=tdata[, features_flag])$posterior[,2]
lda_under_pred <- predict(lda_under, newdata=tdata[, features_flag])$posterior[,2]
lda_original_pred <- predict(lda_original, newdata=tdata[, features_flag])$posterior[,2]

qda_over_pred <- predict(qda_over, newdata=tdata[, features_flag])$posterior[,2]
qda_under_pred <- predict(qda_under, newdata=tdata[, features_flag])$posterior[,2]
qda_original_pred <- predict(qda_original, newdata=tdata[, features_flag])$posterior[,2]

lda_over_pos <- (lda_over_pred >= lda_over_threshold)
total_cost <- 10*sum((lda_over_pos==1)&(test_class==0))+500*sum((lda_over_pos==0)&(test_class==1))

table(lda_over_pos, test_class)

# For oversampled training dataset total_cost = 14,240
#        test_class
# over_pos     0     1
#    FALSE 14801    12
#    TRUE    824   363

lda_under_pos <- (lda_under_pred >= lda_under_threshold)
total_cost <- 10*sum((lda_under_pos==1)&(test_class==0))+500*sum((lda_under_pos==0)&(test_class==1))

table(lda_under_pos, test_class)

# For undersampled training dataset total_cost = 13,620
#         test_class
# under_pos     0     1
#     FALSE 14813    11
#     TRUE    812   364

lda_original_pos <- (lda_original_pred >= lda_original_threshold)
total_cost <- 10*sum((lda_original_pos==1)&(test_class==0))+500*sum((lda_original_pos==0)&(test_class==1))

table(lda_original_pos, test_class)

# For original training dataset total_cost = 22,760
#            test_class
# original_pos     0     1
#        FALSE 14849    30
#        TRUE    776   345

qda_over_pos <- (qda_over_pred >= qda_over_threshold)
total_cost <- 10*sum((qda_over_pos==1)&(test_class==0))+500*sum((qda_over_pos==0)&(test_class==1))

table(qda_over_pos, test_class)

# For oversampled training dataset total_cost = 20,900

#        test_class
# qda_over_pos     0     1
#        FALSE 14435    18
#        TRUE   1190   357

qda_under_pos <- (qda_under_pred >= qda_under_threshold)
total_cost <- 10*sum((qda_under_pos==1)&(test_class==0))+500*sum((qda_under_pos==0)&(test_class==1))
table(qda_under_pos, test_class)
# For undersampled training dataset total_cost = 21,760
#             test_class
# qda_under_pos     0     1
#         FALSE 14249    16
#         TRUE   1376   359
# We've lost the tuning gain, but we could get reasonable result on a much-much smaller dataset!

qda_original_pos <- (qda_original_pred >= qda_original_threshold)
total_cost <- 10*sum((qda_original_pos==1)&(test_class==0))+500*sum((qda_original_pos==0)&(test_class==1))

table(qda_original_pos, test_class)

# For original training dataset total_cost = 21,040
#                test_class
# qda_original_pos     0     1
#            FALSE 14471    19
#            TRUE   1154   356

#############################################################################################

# Naive-Bayes - fitting original, undersampled and oversampled data

library(naivebayes)

nb_over <- naive_bayes(x=ovdata[,-1], y=ovdata[,1])
nb_under <- naive_bayes(x=undata[,-1], y=undata[,1])
nb_original <- naive_bayes(x=sampling_data[,-1], y=sampling_data[,1])

over_fitted_pred <- predict(nb_over, newdata=ovdata[,-1], type="prob")[,2]
under_fitted_pred <- predict(nb_under, newdata=undata[,-1], type="prob")[,2]
original_fitted_pred <- predict(nb_original, newdata=sampling_data[,-1], type="prob")[,2]

threshold <- seq(from=0.001, to=0.6, by=0.001)
total_cost <- matrix(rep(0, 3*length(threshold)),ncol=3)

for(i in 1:length(threshold)){
  over_fitted_pos <- (over_fitted_pred>=threshold[i])
  under_fitted_pos <- (under_fitted_pred>=threshold[i])
  original_fitted_pos <- (original_fitted_pred>=threshold[i])
  total_cost[i,1] <- 10*sum((over_fitted_pos==1)&(ovdata[,1]==0))+500*sum((over_fitted_pos==0)&(ovdata[,1]==1))
  total_cost[i,2] <- 10*sum((under_fitted_pos==1)&(undata[,1]==0))+500*sum((under_fitted_pos==0)&(undata[,1]==1))
  total_cost[i,3] <- 10*sum((original_fitted_pos==1)&(num_class==0))+500*sum((original_fitted_pos==0)&(num_class==1))
}

plot(threshold,total_cost[,3]) # original training data
original_threshold <- threshold[which.min(total_cost[,3])] # 0.49

plot(threshold,total_cost[,1]) # over-sampled training data

over_threshold <- 0.5 # uninformative argmin=0.01 would yield total_cost=27,670

plot(threshold,total_cost[,2]) # under-sampled training data
under_threshold <- 0.5 # uninformative argmin=0.01 would yield total_cost=29,250

#############################################################################################

# Testing NB on test dataset

over_pred <- predict(nb_over, newdata=tdata[, features_flag], type="prob")[,2]
under_pred <- predict(nb_under, newdata=tdata[, features_flag], type="prob")[,2]
original_pred <- predict(nb_original, newdata=tdata[, features_flag], type="prob")[,2]

over_pos <- (over_pred >= over_threshold)
total_cost <- 10*sum((over_pos==1)&(test_class==0))+500*sum((over_pos==0)&(test_class==1))

table(over_pos, test_class)
# For oversampled training dataset total_cost = 24,880
#        test_class
# over_pos    0     1
#   FALSE 14037    18
#   TRUE   1588   357

under_pos <- (under_pred >= under_threshold)
total_cost <- 10*sum((under_pos==1)&(test_class==0))+500*sum((under_pos==0)&(test_class==1))

table(under_pos, test_class)
# For undersampled training dataset total_cost = 26,670
#         test_class
# under_pos    0     1
#    FALSE 13958    20
#    TRUE   1667   355

original_pos <- (original_pred >= original_threshold)
total_cost <- 10*sum((original_pos==1)&(test_class==0))+500*sum((original_pos==0)&(test_class==1))

table(original_pos, test_class)
# For original training dataset total_cost = 24,510
#            test_class
# original_pos    0     1
#       FALSE 14224    21
#       TRUE   1401   354

# We can see that NB works bad on the dataset of heavily codependent features

#############################################################################################

# Decision trees for over/under/original data
# w/o histogram features

library(tree)

# Fitting

tree.over = tree(num_class ~ ., ovdata)
tree.under = tree(num_class ~ ., undata)
tree.original = tree(num_class ~ ., sampling_data)

summary(tree.over)   
plot(tree.over)
text(tree.over)	
tree.over

summary(tree.under)
plot(tree.under)
text(tree.under)	
tree.under

summary(tree.original)
plot(tree.original)
text(tree.original)	
tree.original

over_fitted_pred <- predict(tree.over, newdata=ovdata, type="vector")
under_fitted_pred <- predict(tree.under, newdata=undata, type="vector")
original_fitted_pred <- predict(tree.original, newdata=sampling_data, type="vector")

threshold <- seq(from=0.001, to=0.3, by=0.001)
total_cost <- matrix(rep(0, 3*length(threshold)),ncol=3)

for(i in 1:length(threshold)){
  over_fitted_pos <- (over_fitted_pred>=threshold[i])
  under_fitted_pos <- (under_fitted_pred>=threshold[i])
  original_fitted_pos <- (original_fitted_pred>=threshold[i])
  total_cost[i,1] <- 10*sum((over_fitted_pos==1)&(ovdata[,1]==0))+500*sum((over_fitted_pos==0)&(ovdata[,1]==1))
  total_cost[i,2] <- 10*sum((under_fitted_pos==1)&(undata[,1]==0))+500*sum((under_fitted_pos==0)&(undata[,1]==1))
  total_cost[i,3] <- 10*sum((original_fitted_pos==1)&(num_class==0))+500*sum((original_fitted_pos==0)&(num_class==1))
}

plot(threshold,total_cost[,3]) # original training data
original_threshold <- threshold[which.min(total_cost[,3])] # 0.003

plot(threshold,total_cost[,1]) # over-sampled training data
over_threshold <- 0.5 #threshold[which.min(total_cost[,1])] # 0.001

plot(threshold,total_cost[,2]) # under-sampled training data
under_threshold <- 0.5 #threshold[which.min(total_cost[,2])] # 0.001

#############################################################################################

# Testing decision trees 

over_pred <- predict(tree.over, newdata=tdata[, features_flag], type="vector")
under_pred <- predict(tree.under, newdata=tdata[, features_flag], type="vector")
original_pred <- predict(tree.original, newdata=tdata[, features_flag], type="vector")

over_pos <- (over_pred >= over_threshold)
total_cost <- 10*sum((over_pos==1)&(test_class==0))+500*sum((over_pos==0)&(test_class==1))

table(over_pos, test_class)
# For oversampled training dataset total_cost = 18,640
#        test_class
# over_pos    0     1
#   FALSE 14761    20
#   TRUE    864   355

under_pos <- (under_pred >= under_threshold)
total_cost <- 10*sum((under_pos==1)&(test_class==0))+500*sum((under_pos==0)&(test_class==1))

table(under_pos, test_class)
# For undersampled training dataset total_cost = 18,790
#         test_class
# under_pos    0     1
#    FALSE 14746    20
#    TRUE    879   355

original_pos <- (original_pred >= original_threshold)
total_cost <- 10*sum((original_pos==1)&(test_class==0))+500*sum((original_pos==0)&(test_class==1))

table(original_pos, test_class)
# For original training dataset total_cost = 25,270
#            test_class
# original_pos    0     1
#       FALSE 15198    42
#       TRUE    427   333

#############################################################################################
# Applying C.5.0 tree-base algorithm on over/under/original datasets with no fitting

library(caret)

# 5-fold CV repeated 5 times
fitControl <- trainControl(method="repeatedcv", number=5, repeats=5, returnResamp="all")
grid <- expand.grid(.winnow=c(TRUE,FALSE), .trials=c(1,5,10,15,20), .model="tree")

x <- sampling_data[,-1]
y <- as.factor(sampling_data[,1])
original_c50 <- train(x=x, y=y, tuneGrid=grid, trControl=fitControl, method="C5.0", verbose=FALSE)
original_c50_pred <- predict(original_c50, tdata[,features_flag])
table(original_c50_pred, test_class)
total_cost <- sum(10*((original_c50_pred=='1')&(test_class==0))+500*((original_c50_pred=='0')&(test_class==1)))
total_cost
# For original training dataset total_cost = 81,760 around 4 hours
#                 test_class
# original_c50_pred     0     1
#                 0 15599   163
#                 1    26   212

x <- undata[,-1]
y <- as.factor(undata[,1])
under_c50 <- train(x=x, y=y, tuneGrid=grid, trControl=fitControl, method="C5.0", verbose=FALSE)
under_c50_pred <- predict(under_c50, tdata[,features_flag])
table(under_c50_pred, test_class)
total_cost <- sum(10*((under_c50_pred=='1')&(test_class==0))+500*((under_c50_pred=='0')&(test_class==1)))
total_cost
# For undersampled training dataset total_cost = 12,150, around 6 minutes
#              test_class
# under_c50_pred     0     1
#              0 14860     9
#              1   765   366

x <- ovdata[,-1]
y <- as.factor(ovdata[,1])
set.seed(1)
sub <- sample(1:nrow(ovdata), 20000)
over_c50 <- train(x=x[sub,], y=y[sub], tuneGrid=grid, trControl=fitControl, method="C5.0", verbose=FALSE)
over_c50_pred <- predict(over_c50, tdata[,features_flag])
table(over_c50_pred, test_class)
total_cost <- sum(10*((over_c50_pred=='1')&(test_class==0))+500*((over_c50_pred=='0')&(test_class==1)))
total_cost
# For oversampled training dataset total_cost = 23,060 about 50 minutes
# 20,000 subset of over-sampled data
#             test_class
# over_c50_pred     0     1
#              0 15369    41
#              1   256   334

# for 60,000 subset of over-sampled data total cost = 36,280
#             test_class
# over_c50_pred     0     1
#             0 15497    70
#             1   128   305

#############################################################################################
# Fitting C.5.0 with the probabilities threshold on over/under/original data

fitControl <- trainControl(method="repeatedcv", number=5, repeats=5, returnResamp="all", classProbs=TRUE)
grid <- expand.grid(.winnow=c(TRUE,FALSE), .trials=c(1,5,10,15,20), .model="tree")

x <- undata[,-1]
y <- ifelse(undata[,1]==1, 'pos', 'neg')
under_c50 <- train(x=x, y=y, tuneGrid=grid, trControl=fitControl, method="C5.0", verbose=FALSE)
under_c50_fit <- predict(under_c50, newdata=x, type="prob")[,2]

threshold <- seq(from=0, to=0.9, by=0.01)
total_cost <- rep(0, length(threshold))
for (i in 1:length(threshold)){
  under_c50_pos <- (under_c50_fit>=threshold[i])
  total_cost[i] <- sum(10*((under_c50_pos==1)&(undata[,1]==0))+500*((under_c50_pos==0)&(undata[,1]==1)))
}
plot(total_cost)
under_c50_threshold <- threshold[which.min(total_cost)] # 0.24

under_c50_pred <- predict(under_c50, tdata[,features_flag], type="prob")[,2]
under_c50_pos <- (under_c50_pred>=under_c50_threshold)
table(under_c50_pos, test_class)
total_cost <- sum(10*((under_c50_pos==1)&(test_class==0))+500*((under_c50_pos==0)&(test_class==1)))
total_cost
# For undersampled training dataset total_cost = 17,730, around 6 minutes
#              test_class
# under_c50_pos     0     1
#         FALSE 13952     2
#         TRUE   1673   373

x <- sampling_data[,-1]
y <- ifelse(sampling_data[,1]==1, 'pos', 'neg')
original_c50 <- train(x=x, y=y, tuneGrid=grid, trControl=fitControl, method="C5.0", verbose=FALSE)
original_c50_fit <- predict(original_c50, newdata=x, type="prob")[,2]
threshold <- seq(from=0, to=0.9, by=0.01)
total_cost <- rep(0, length(threshold))
for (i in 1:length(threshold)){
  original_c50_pos <- (original_c50_fit>=threshold[i])
  total_cost[i] <- sum(10*((original_c50_pos==1)&(sampling_data[,1]==0))+500*((original_c50_pos==0)&(sampling_data[,1]==1)))
}
plot(total_cost)
original_c50_threshold <- threshold[which.min(total_cost)] # 0.21
# Note that for this trees based method optimal threshold is roughly the same for balanced and unbalanced data
original_c50_threshold

original_c50_pred <- predict(original_c50, tdata[,features_flag], type="prob")[2]
original_c50_pos <- (original_c50_pred>=original_c50_threshold)
table(original_c50_pos, test_class)
total_cost <- sum(10*((original_c50_pos==1)&(test_class==0))+500*((original_c50_pos==0)&(test_class==1)))
total_cost
# For original training dataset total_cost = 25,850, around 5 hours
#                test_class
# original_c50_pos     0     1
#            FALSE 15440    48
#            TRUE    185   327

#############################################################################################
# Support Vector Machines (SVM) for over/under/original data (w/o histogram features)

# Part 1: Linear Kernel
# Subsets of 2000 elements of training sets (oversampled and original) used for fitting the models

library(e1071)
#############################################################################################
# TUNNING

# 1. undersample
set.seed(1)
tune.out <- tune(svm, num_class~., data=undata, kernel="linear", ranges=list(cost=c(0.01,0.1,1,10)))
# RUNNING TIME ~5 min
summary(tune.out)
#  cost      error  dispersion
#1  0.01 0.06281091 0.005313371
#2  0.10 0.06178601 0.005635465
#3  1.00 0.06181925 0.005591241
#4 10.00 0.06199293 0.005632159
svm.model <- tune.out$best.model
summary(svm.model)
# optimal cost = 0.1 (linear kernel, undersample)

# 2. oversample - using subset of 2000 instances
set.seed(1)
sample <- sample(1:dim(ovdata)[1],2000)
ovdata_sample <- ovdata[sample,]
tune.out <- tune(svm, num_class~., data=ovdata_sample, kernel="linear", ranges=list(cost=c(0.01,0.1,1,10)))
# RUNNING TIME ~8 min
summary(tune.out)
#  cost      error dispersion
#1  0.01 0.05983663 0.01289405
#2  0.10 0.05955815 0.01305385
#3  1.00 0.05963987 0.01296555
#4 10.00 0.05953335 0.01299092
svm.model <- tune.out$best.model
summary(svm.model)
# optimal cost = 10 (linear kernel, oversample)

# 3. original sample
set.seed(1)
sample <- sample(1:dim(sampling_data)[1],2000)
sampling_data_sample <- sampling_data[sample,]
tune.out <- tune(svm, num_class~., data=sampling_data_sample, kernel="linear", ranges=list(cost=c(0.01,0.1,1,10)))
# RUNNING TIME ~8 min
summary(tune.out)
#  cost      error dispersion
#1  0.01 0.01883876 0.01449081
#2  0.10 0.01873112 0.01440458
#3  1.00 0.01872288 0.01437944
#4 10.00 0.01871932 0.01438601
svm.model <- tune.out$best.model
summary(svm.model)
# optimal cost = 10 (linear kernel, original data)

#############################################################################################
# FITTING THE DATA AND FINDING OPTIMAL THRESHOLD FOR ORIGINAL DATA
# TRAINING OF FULL UNDERSAMPLE AND SUBSETS OF 2000 ELEMENTS FROM OVERSAMPLE AND ORIGINAL DATASET

svm.model_under = svm(num_class~., data=undata, kernel ="linear", cost=0.1, scale = T)
svm.model_over = svm(num_class~., data=ovdata_sample, kernel ="linear", cost=10, scale = T)
svm.model_original = svm(num_class~., data=sampling_data_sample, kernel ="linear", cost=10, scale = T)

summary(svm.model_over)
summary(svm.model_under)
summary(svm.model_original)

over_fitted_pred <- predict(svm.model_over, newdata=ovdata)
under_fitted_pred <- predict(svm.model_under, newdata=undata)
original_fitted_pred <- predict(svm.model_original, newdata=sampling_data)

threshold <- seq(from=0.001, to=0.5, by=0.001)
total_cost <- matrix(rep(0, 3*length(threshold)),ncol=3)

for(i in 1:length(threshold)){
  over_fitted_pos <- (over_fitted_pred>=threshold[i])
  under_fitted_pos <- (under_fitted_pred>=threshold[i])
  original_fitted_pos <- (original_fitted_pred>=threshold[i])
  total_cost[i,1] <- 10*sum((over_fitted_pos==1)&(ovdata[,1]==0))+500*sum((over_fitted_pos==0)&(ovdata[,1]==1))
  total_cost[i,2] <- 10*sum((under_fitted_pos==1)&(undata[,1]==0))+500*sum((under_fitted_pos==0)&(undata[,1]==1))
  total_cost[i,3] <- 10*sum((original_fitted_pos==1)&(num_class==0))+500*sum((original_fitted_pos==0)&(num_class==1))
}

plot(threshold,total_cost[,3]) # original training data
original_threshold <- threshold[which.min(total_cost[,3])] # optimal threshold = 0.012

plot(threshold,total_cost[,1]) # over-sampled training data
over_threshold <- 0.5 #threshold[which.min(total_cost[,1])] # optimal threshold = 0.001

plot(threshold,total_cost[,2]) # under-sampled training data
under_threshold <- 0.5 #threshold[which.min(total_cost[,2])] # optimal threshold = 0.165

#############################################################################################
# SVM TESTING (KERNEL = LINEAR)

over_pred <- predict(svm.model_over, newdata=tdata[, features_flag])
under_pred <- predict(svm.model_under, newdata=tdata[, features_flag])
original_pred <- predict(svm.model_original, newdata=tdata[, features_flag])

over_pos <- (over_pred >= over_threshold)
total_cost <- 10*sum((over_pos==1)&(test_class==0))+500*sum((over_pos==0)&(test_class==1))
total_cost
table(over_pos, test_class)
# For oversampled training dataset total_cost = 16,460
#           test_class
# over_pos     0     1
# FALSE     14729    15
# TRUE        896   360

under_pos <- (under_pred >= under_threshold)
total_cost <- 10*sum((under_pos==1)&(test_class==0))+500*sum((under_pos==0)&(test_class==1))
total_cost
table(under_pos, test_class)
# For undersampled training dataset total_cost = 15,210
#        test_class
# under_pos   0     1
#   FALSE 14654    11
#   TRUE    971   364

original_pos <- (original_pred >= original_threshold)
total_cost <- 10*sum((original_pos==1)&(test_class==0))+500*sum((original_pos==0)&(test_class==1))
total_cost
table(original_pos, test_class)
# For original training dataset total_cost = 51,450
#               test_class
# original_pos    0       1
# FALSE           14480   80
# TRUE            1145    295

#############################################################################################
# Support Vector Machines (SVM) for over/under/original data (w/o histogram features)
# Part 2: Radial Kernel
# Subsets of 2000 elements of training sets (oversampled and original) used for fitting the models
library(e1071)

#############################################################################################
# TUNNING

# 1. undersample
set.seed(1)
tune.out <- tune(svm, num_class~., data=undata, kernel="radial", ranges=list(cost=c(0.01,0.1,1,10)))
# RUNNING TIME ~5 min
summary(tune.out)
#        cost      error  dispersion
#1  0.01 0.08723201 0.008462292
#2  0.10 0.06016571 0.008000008
#3  1.00 0.04762150 0.007998566
#4 10.00 0.05128376 0.008342889
svm.model <- tune.out$best.model
summary(svm.model)
# optimal cost = 1 (radial kernel, undersample)

# 2. oversample - using subset of 2000 instances
set.seed(1)
sample <- sample(1:dim(ovdata)[1],2000)
ovdata_sample <- ovdata[sample,]
tune.out <- tune(svm, num_class~., data=ovdata_sample, kernel="radial", ranges=list(cost=c(0.01,0.1,1,10)))
# RUNNING TIME ~3 min
summary(tune.out)
#        cost      error  dispersion
#1  0.01 0.08054704 0.012508391
#2  0.10 0.05514016 0.013978262
#3  1.00 0.03901209 0.011797421
#4 10.00 0.03597675 0.007662162
svm.model <- tune.out$best.model
summary(svm.model)
#optimal cost = 10 (radial kernel, oversample)

# 3. original sample - using subset of 2000 instances
set.seed(1)
sample <- sample(1:dim(sampling_data)[1],2000)
sampling_data_sample <- sampling_data[sample,]
tune.out <- tune(svm, num_class~., data=sampling_data_sample, kernel="radial", ranges=list(cost=c(0.01,0.1,1,10)))
# RUNNING TIME ~3 min
summary(tune.out)
#        cost      error  dispersion
#1  0.01 0.01962105 0.014864380
#2  0.10 0.01934678 0.014748666
#3  1.00 0.01749630 0.013599803
#4 10.00 0.01216614 0.008013373
svm.model <- tune.out$best.model
summary(svm.model)
# optimal cost = 10 (radial kernel, original data)

#############################################################################################
# FITTING THE DATA AND FINDING OPTIMAL THRESHOLD FOR ORIGINAL DATA
# TRAINING OF FULL UNDERSAMPLE AND SUBSETS OF 2000 ELEMENTS FROM OVERSAMPLE AND ORIGINAL DATASET

svm.model_under = svm(num_class~., data=undata, kernel ="radial", cost=1, scale = T)
svm.model_over = svm(num_class~., data=ovdata_sample, kernel ="radial", cost=10, scale = T)
svm.model_original = svm(num_class~., data=sampling_data_sample, kernel ="radial", cost=10, scale = T)

summary(svm.model_over)
summary(svm.model_under)
summary(svm.model_original)

over_fitted_pred <- predict(svm.model_over, newdata=ovdata)
under_fitted_pred <- predict(svm.model_under, newdata=undata)
original_fitted_pred <- predict(svm.model_original, newdata=sampling_data)

threshold <- seq(from=0.001, to=0.5, by=0.001)
total_cost <- matrix(rep(0, 3*length(threshold)),ncol=3)

for(i in 1:length(threshold)){
  over_fitted_pos <- (over_fitted_pred>=threshold[i])
  under_fitted_pos <- (under_fitted_pred>=threshold[i])
  original_fitted_pos <- (original_fitted_pred>=threshold[i])
  total_cost[i,1] <- 10*sum((over_fitted_pos==1)&(ovdata[,1]==0))+500*sum((over_fitted_pos==0)&(ovdata[,1]==1))
  total_cost[i,2] <- 10*sum((under_fitted_pos==1)&(undata[,1]==0))+500*sum((under_fitted_pos==0)&(undata[,1]==1))
  total_cost[i,3] <- 10*sum((original_fitted_pos==1)&(num_class==0))+500*sum((original_fitted_pos==0)&(num_class==1))
}

plot(threshold,total_cost[,3]) # original training data
original_threshold <- threshold[which.min(total_cost[,3])] # optimal threshold = 0.068

plot(threshold,total_cost[,1]) # over-sampled training data
over_threshold <- 0.5 #threshold[which.min(total_cost[,1])] # optimal threshold = 0.157

plot(threshold,total_cost[,2]) # under-sampled training data
under_threshold <- 0.5 #threshold[which.min(total_cost[,2])] # optimal threshold = 0.063

#############################################################################################
# SVM TESTING (KERNEL = RADIAL)

over_pred <- predict(svm.model_over, newdata=tdata[, features_flag])
under_pred <- predict(svm.model_under, newdata=tdata[, features_flag])
original_pred <- predict(svm.model_original, newdata=tdata[, features_flag])

over_pos <- (over_pred >= over_threshold)
total_cost <- 10*sum((over_pos==1)&(test_class==0))+500*sum((over_pos==0)&(test_class==1))
total_cost
table(over_pos, test_class)
# For oversampled training dataset total_cost = 19,460
#           test_class
# over_pos     0     1
# FALSE     14929    25
# TRUE        696   350

# For undersampled training dataset total_cost = 16,940
under_pos <- (under_pred >= under_threshold)
total_cost <- 10*sum((under_pos==1)&(test_class==0))+500*sum((under_pos==0)&(test_class==1))
total_cost
table(under_pos, test_class)
#         test_class
# under_pos   0     1
#   FALSE 14681    15
#   TRUE    944   360

# For original training dataset total_cost = 23,020
original_pos <- (original_pred >= original_threshold)
total_cost <- 10*sum((original_pos==1)&(test_class==0))+500*sum((original_pos==0)&(test_class==1))
total_cost
table(original_pos, test_class)
#               test_class
#original_pos     0     1
#FALSE           14523  24
#TRUE            1102   351

#############################################################################################
# Support Vector Machines (SVM) for over/under/original data (w/o histogram features)
# Part 3: Polynomial Kernel
# Subsets of 2000 elements of training sets (oversampled and original) used for fitting the models

library(e1071)
#############################################################################################
# TUNNING

# 1. undersample
set.seed(1)
tune.out <- tune(svm, num_class~., data=undata, kernel="polynomial", ranges=list(cost=c(0.01,0.1,1,10)))
summary(tune.out)
#        cost      error  dispersion
#1  0.01 0.20301743 0.016683502
#2  0.10 0.09205724 0.005488299
#3  1.00 0.06957129 0.012616181
#4 10.00 0.07740779 0.015901190
svm.model <- tune.out$best.model
summary(svm.model)
# optimal cost = 1 (polynomial kernel, undersample)

# 2. oversample - using subset of 2000 instances
set.seed(1)
sample <- sample(1:dim(ovdata)[1],2000)
ovdata_sample <- ovdata[sample,]
tune.out <- tune(svm, num_class~., data=ovdata_sample, kernel="polynomial", ranges=list(cost=c(0.01,0.1,1,10)))
# RUNNING TIME ~3 min
summary(tune.out)
#        cost      error  dispersion
#1  0.01 0.16369785 0.007873918
#2  0.10 0.08255612 0.013513949
#3  1.00 0.04982746 0.009489707
#4 10.00 0.04780833 0.012188309
svm.model <- tune.out$best.model
summary(svm.model)
# optimal cost = 10 (polynomial kernel, oversample)

# 3. original sample - using subset of 2000 instances
set.seed(1)
sample <- sample(1:dim(sampling_data)[1],2000)
sampling_data_sample <- sampling_data[sample,]
tune.out <- tune(svm, num_class~., data=sampling_data_sample, kernel="polynomial", ranges=list(cost=c(0.01,0.1,1,10)))
summary(tune.out)
#- Detailed performance results:
#        cost      error  dispersion
#1  0.01 0.01752912 0.014252663
#2  0.10 0.01438677 0.011683703
#3  1.00 0.01170891 0.008415986
#4 10.00 0.01206959 0.007949114
svm.model <- tune.out$best.model
summary(svm.model)
# optimal cost = 1 (polynomial kernel, original data)

#############################################################################################
# FITTING THE DATA AND FINDING OPTIMAL THRESHOLD FOR ORIGINAL DATA
# TRAINING OF FULL UNDERSAMPLE AND SUBSETS OF 2000 ELEMENTS FROM OVERSAMPLE AND ORIGINAL DATASET

svm.model_under = svm(num_class~., data=undata, kernel ="polynomial", cost=1, scale = T)
svm.model_over = svm(num_class~., data=ovdata_sample, kernel ="polynomial", cost=10, scale = T)
svm.model_original = svm(num_class~., data=sampling_data_sample, kernel ="polynomial", cost=1, scale = T)

summary(svm.model_over)
summary(svm.model_under)
summary(svm.model_original)

over_fitted_pred <- predict(svm.model_over, newdata=ovdata)
under_fitted_pred <- predict(svm.model_under, newdata=undata)
original_fitted_pred <- predict(svm.model_original, newdata=sampling_data)

threshold <- seq(from=0.001, to=0.5, by=0.001)
total_cost <- matrix(rep(0, 3*length(threshold)),ncol=3)

for(i in 1:length(threshold)){
  over_fitted_pos <- (over_fitted_pred>=threshold[i])
  under_fitted_pos <- (under_fitted_pred>=threshold[i])
  original_fitted_pos <- (original_fitted_pred>=threshold[i])
  total_cost[i,1] <- 10*sum((over_fitted_pos==1)&(ovdata[,1]==0))+500*sum((over_fitted_pos==0)&(ovdata[,1]==1))
  total_cost[i,2] <- 10*sum((under_fitted_pos==1)&(undata[,1]==0))+500*sum((under_fitted_pos==0)&(undata[,1]==1))
  total_cost[i,3] <- 10*sum((original_fitted_pos==1)&(num_class==0))+500*sum((original_fitted_pos==0)&(num_class==1))
}

plot(threshold,total_cost[,3]) # original training data
original_threshold <- threshold[which.min(total_cost[,3])] # optimal threshold = 0.026

plot(threshold,total_cost[,1]) # over-sampled training data
over_threshold <- 0.5 #threshold[which.min(total_cost[,1])] # optimal threshold = 0.072

plot(threshold,total_cost[,2]) # under-sampled training data
under_threshold <- 0.5 #threshold[which.min(total_cost[,2])] # optimal threshold = 0.058

#############################################################################################
# SVM TESTING (KERNEL = POLYNOMIAL)

over_pred <- predict(svm.model_over, newdata=tdata[, features_flag])
under_pred <- predict(svm.model_under, newdata=tdata[, features_flag])
original_pred <- predict(svm.model_original, newdata=tdata[, features_flag])

over_pos <- (over_pred >= over_threshold)
total_cost <- 10*sum((over_pos==1)&(test_class==0))+500*sum((over_pos==0)&(test_class==1))
total_cost
table(over_pos, test_class)
# For oversampled training dataset total_cost = 23,890
#          test_class
# over_pos     0     1
# FALSE     14986    35
# TRUE        639   340

under_pos <- (under_pred >= under_threshold)
total_cost <- 10*sum((under_pos==1)&(test_class==0))+500*sum((under_pos==0)&(test_class==1))
total_cost
table(under_pos, test_class)
# For undersampled training dataset total_cost = 21,250
#         test_class
# under_pos   0     1
#   FALSE 14850    27
#   TRUE    775   348

# For original training dataset total_cost = 25,330
original_pos <- (original_pred >= original_threshold)
total_cost <- 10*sum((original_pos==1)&(test_class==0))+500*sum((original_pos==0)&(test_class==1))
total_cost
table(original_pos, test_class)
#              test_class
# original_pos     0     1
# FALSE           14592  30
# TRUE            1033   345

##############################################################################################
#SVM RESULTS (total costs on test set):

# Linear kernel, oversampled data: 16,460
# Linear kernel, undersampled data: 15,210
# Linear kernel, original data: 51,450

# Radial kernel, oversampled data: 19,460
# Radial kernel, undersampled data: 16,940
# Radial kernel, original data: 23,020

# Polynomial kernel, oversampled data: 23,890
# Polynomial kernel, undersampled data: 21,250
# Polynomial kernel, original data: 25,330

##############################################################################################
# RANDOM FORESTS - FULL ORIGINAL DATA (155 Features)

library(randomForest)

# Fitting
p = dim(data[,-(1:2)])[2]                         # p= 155, p/3 = 51 features considered for random forest (mtry)
fulldata = cbind(num_class, data[,-(1:2)])

rf.originalreg = randomForest(num_class ~ ., fulldata, ntree=500, mtry = 51)
# Run time about 40 minutes (with 1000 ntree never completed)

rf.fitted_pred <- predict(rf.originalreg, newdata=fulldata[,-1], type="response")
range(rf.fitted_pred)

threshold <- seq(from=0.005, to=1, by=0.01)
total_cost <- matrix(rep(0, 1*length(threshold)),ncol=1)
for(i in 1:length(threshold)){  
  rf.fitted_pos <- (rf.fitted_pred>=threshold[i])
  total_cost[i,1] <- 10*sum((rf.fitted_pos==1)&(num_class==0))+500*sum((rf.fitted_pos==0)&(num_class==1))
}
rf.original_threshold <- threshold[which.min(total_cost[,1])] # 0.374

#############################################################################################
# Testing random forest - 

testdata = cbind(test_class, tdata[,-(1:2)])
original_pred <- predict(rf.originalreg, newdata=testdata[,-1], type="response")

original_pos <- (original_pred >= rf.original_threshold)
total_cost <- 10*sum((original_pos==1)&(test_class==0))+500*sum((original_pos==0)&(test_class==1))

table(original_pos, test_class)
# For full dataset total_cost =  38,300   
#              test_class
# original_pos     0     1
#   FALSE       15575    76
#  TRUE           30   299

#############################################################################################
# Random Forest with Cross validation technique using CARET package

# 5 fold Cross-Validation with 3 repetitions
control <- trainControl(method="repeatedcv", number=5, repeats=3)
seed <- 7
metric <- "Accuracy"
set.seed(seed)
mtry <- (ncol(undata[,-1])/3)
tunegrid <- expand.grid(.mtry=mtry)

# Undersampled data - fixed value for mtry

x <- undata[,-1]
y <- as.factor(undata[,1])
rf_undata <- train(x=x,y=y, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_undata)
#Accuracy      Kappa       
#0.9529153901  0.9057315655


# Undersampled data - Random Search to get best value for mtry

control <- trainControl(method="repeatedcv", number=5, repeats=3, search="random")
set.seed(seed)
rf_random <- train(x,y, method="rf", metric=metric, tuneLength=15, trControl=control)
print(rf_random)
plot(rf_random)
#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 39.
# mtry  Accuracy      Kappa
# 39  0.9535947967  0.9070906353

# prediction on test data with fixed mtry=28

rf_undata_pred <- predict(rf_undata, tdata[,features_flag])
table(rf_undata_pred, test_class)
#                 test_class
#rf_undata_pred     0     1
#           0   14742     5
#           1    883   370
total_cost <- sum(10*((rf_undata_pred=='1')&(test_class==0))+500*((rf_undata_pred=='0')&(test_class==1)))
total_cost      #11,330

# prediction on test data with best mtry=39 after random search

rf_undata_pred_rand <- predict(rf_random, tdata[,features_flag])
table(rf_undata_pred_rand, test_class)
#                       test_class
#rf_undata_pred_rand      0     1
#                 0   14747     5
#                 1     878   370
total_cost <- sum(10*((rf_undata_pred_rand=='1')&(test_class==0))+500*((rf_undata_pred_rand=='0')&(test_class==1)))
total_cost      #11,280


#############################################################################################
#5 fold CV on original data
control <- trainControl(method="repeatedcv", number=5, repeats=3)
seed <- 7
metric <- "Accuracy"
set.seed(seed)
mtry <- (ncol(undata[,-1])/3)
tunegrid <- expand.grid(.mtry=mtry)

xs <- sampling_data[,-1]
ys <- as.factor(sampling_data[,1])

rf_original <- train(x=xs,y=ys, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_original)
# mtry  Accuracy      Kappa       
# 28  0.9930111111  0.7588827773

rf_original_prob <- predict(rf_original, tdata[,features_flag])
table(rf_original_prob, test_class)
#                 test_class
#rf_original_prob     0     1
#             0   15609   113
#            1     16   262
total_cost <- sum(10*((rf_original_prob=='1')&(test_class==0))+500*((rf_original_prob=='0')&(test_class==1)))
total_cost      #56,660


#############################################################################################
# Fitting RF with the probabilities threshold on under/original data

# under sampled data
fitControl <- trainControl(method="repeatedcv", number=5, repeats=5, returnResamp="all", classProbs=TRUE)
x <- undata[,-1]
y <- ifelse(undata[,1]==1, 'pos', 'neg')
under_rf <- train(x=x,y=y, method="rf", tuneGrid=tunegrid, trControl=fitControl)
under_rf_fit <- predict(under_rf, newdata=x, type="prob")[,2]

threshold <- seq(from=0, to=0.9, by=0.01)
total_cost <- rep(0, length(threshold))
for (i in 1:length(threshold)){
  under_rf_pos <- (under_rf_fit>=threshold[i])
  total_cost[i] <- sum(10*((under_rf_pos==1)&(undata[,1]==0))+500*((under_rf_pos==0)&(undata[,1]==1)))
}
plot(total_cost)
under_rf_threshold <- threshold[which.min(total_cost)] # 0.4

under_rf_pred <- predict(under_rf, tdata[,features_flag], type="prob")[,2]
under_rf_pos <- (under_rf_pred>=under_rf_threshold)
table(under_rf_pos, test_class)
total_cost <- sum(10*((under_rf_pos==1)&(test_class==0))+500*((under_rf_pos==0)&(test_class==1)))
total_cost
# For undersampled training dataset total_cost = 12,750
#              test_class
# under_rf_pos     0     1
#         FALSE 14550     4
#         TRUE   1075   371

# original data
fitControl <- trainControl(method="repeatedcv", number=5, repeats=3, returnResamp="all", classProbs=TRUE)
xs <- sampling_data[,-1]
ys <- ifelse(sampling_data[,1]==1, 'pos', 'neg')
tic() 
original_rf <- train(x=xs, y=ys, method="rf", tuneGrid=tunegrid, trControl=fitControl,  verbose=FALSE)
toc()
#24403.18 sec elapsed
print(original_rf)
original_rf_fit <- predict(original_rf, newdata=xs, type="prob")[,2]
#Accuracy      Kappa       
#0.9930111111  0.7593540004

threshold <- seq(from=0, to=0.5, by=0.01)
total_cost <- rep(0, length(threshold))
for (i in 1:length(threshold)){
  original_rf_pos <- (original_rf_fit>=threshold[i])
  total_cost[i] <- sum(10*((original_rf_pos==1)&(sampling_data[,1]==0))+500*((original_rf_pos==0)&(sampling_data[,1]==1)))
}
plot(total_cost)
original_rf_threshold <- threshold[which.min(total_cost)] # 0.33
# Note that for this trees based method optimal threshold is roughly the same for balanced and unbalanced data
original_rf_threshold

original_rf_pred <- predict(original_rf, tdata[,features_flag], type="prob")[2]
original_rf_pos <- (original_rf_pred>=original_rf_threshold)
table(original_rf_pos, test_class)
total_cost <- sum(10*((original_rf_pos==1)&(test_class==0))+500*((original_rf_pos==0)&(test_class==1)))
total_cost
# For original training dataset total_cost =  32130
#                test_class
# original_rf_pos     0     1
#            FALSE 15562    63
#            TRUE    63   312 


#############################################################################################
# Neural Network

library(neuralnet)
set.seed(12)

# Basic neural network model training with sampling_data with histogram variables (86 features)

nn.sampling_output = neuralnet(num_class ~ aa_000 + ac_000 + ad_000 + ae_000 +ai_000 +aj_000+ ak_000+ am_0 + ar_000 + at_000 +	 av_000 	+ ax_000 +bc_000 + bd_000 + be_000 + bf_000 + bi_000 + bk_000+bl_000 + bm_000 + bn_000 + bo_000 + bp_000+ bq_000 +br_000 + bs_000 + bu_000 + by_000 + bz_000 + ca_000 +	 cb_000 	+ ce_000+ cf_000 + cg_000 +cj_000 + ck_000 +cl_000 +cm_000+ co_000 + cp_000 + cu_000 + cx_000+cy_000+ cz_000 + da_000 +db_000 +dc_000 + dd_000 + de_000 + df_000 + dg_000 +dh_000 + di_000 + do_000 + dp_000 + dq_000 + dr_000 +dt_000+ du_000 + dv_000 + dx_000 +	 dy_000 + eb_000 + ec_00 + ed_000 + group2_na + group3_na + hist_na 	+ ag_m 	+ ay_m 	 +az_m+ ba_m 	+ cn_m 	 +cs_m+ ee_m + ag_f + ay_f + az_f + ba_f + cn_f + cs_f +ee_f + group2500_na +	 group2723_na + group4800_na, data= sampling_data)

nn.sampling_output_un = neuralnet(num_class ~ aa_000 + ac_000 + ad_000 + ae_000 +ai_000 +aj_000+ ak_000+ am_0 + ar_000 + at_000 +	 av_000 	+ ax_000 +bc_000 + bd_000 + be_000 + bf_000 + bi_000 + bk_000+bl_000 + bm_000 + bn_000 + bo_000 + bp_000+ bq_000 +br_000 + bs_000 + bu_000 + by_000 + bz_000 + ca_000 +	 cb_000 	+ ce_000+ cf_000 + cg_000 +cj_000 + ck_000 +cl_000 +cm_000+ co_000 + cp_000 + cu_000 + cx_000+cy_000+ cz_000 + da_000 +db_000 +dc_000 + dd_000 + de_000 + df_000 + dg_000 +dh_000 + di_000 + do_000 + dp_000 + dq_000 + dr_000 +dt_000+ du_000 + dv_000 + dx_000 +	 dy_000 + eb_000 + ec_00 + ed_000 + group2_na + group3_na + hist_na 	+ ag_m 	+ ay_m 	 +az_m+ ba_m 	+ cn_m 	 +cs_m+ ee_m + ag_f + ay_f + az_f + ba_f + cn_f + cs_f +ee_f + group2500_na +	 group2723_na + group4800_na, data= undata)

# predict on training data sets
nn.train = compute(nn.sampling_output, sampling_data[,-1])
nn.un.train = compute(nn.sampling_output, undata[,-1])

# Testing neural network on test data

# using sampled data
nn.test = compute(nn.sampling_output, tdata[, features_flag])
nn.test_sse = sum((nn.test$net.result - test_class)^2)/2      #test error=183.7971068  

#using undersampled data - 1 hidden layer
nn.test_un = compute(nn.sampling_output_un, tdata[, features_flag])
nn.test_un_sse = sum((nn.test_un$net.result - test_class)^2)/2      #test error =498.5236039

#Accuracy
threshold <- seq(from=0.001, to=.03, by=0.001)
total_cost <- matrix(rep(0, 2*length(threshold)),ncol=2)

for(i in 1:length(threshold)){
  
  fitted_pos <- (nn.train$net.result>=threshold[i])
  un_fitted_pos <- (nn.un.train$net.result>=threshold[i])
  total_cost[i,1] <- 10*sum((fitted_pos==1)&(num_class==0))+500*sum((fitted_pos==0)&(num_class==1))
  total_cost[i,2] <- 10*sum((un_fitted_pos==1)&(undata[,1]==0))+500*sum((un_fitted_pos==0)&(undata[,1]==1))
}                          
original_threshold <- threshold[which.min(total_cost[,1])] # 0.01
under_threshold <- 0.5

# For sampling_data training dataset total_cost = 156,230
original_pos <- (nn.test$net.result >= original_threshold)
total_cost <- 10*sum((original_pos==1)&(test_class==0))+500*sum((original_pos==0)&(test_class==1))
table(original_pos,test_class)
#               test_class
#original_pos     0     1
#FALSE            2     0
#TRUE           15623   375

# For undersampled training dataset total_cost = 18,170
under_pos <- (nn.test_un$net.result >= under_threshold)
total_cost <- 10*sum((under_pos==1)&(test_class==0))+500*sum((under_pos==0)&(test_class==1))
table(under_pos,test_class)
#               test_class
#under_pos     0     1
#FALSE      14558    15
#TRUE       1067   360

#############################################################################################
# STACKING

undata1 =undata
undata1$num_class <- ifelse(undata1$num_class==1, "pos","neg")
test_class_st <- ifelse(test_class==1, "pos","neg")

# create submodels - using under sampled data
fitControl <- trainControl(method="repeatedcv", number =5, repeats=3, savePredictions ="final", classProbs=TRUE)
tic('model_list') 
set.ssed(20)
model_list <- caretList(num_class ~ . , data= undata1,trControl = fitControl, methodList =c("glm", "lda", "rf","knn"))
model_list_1 <- caretList(num_class ~ . , data= undata1,trControl = fitControl, methodList =c("rpart", "lda", "rf","knn"))

toc()
# check correlation and results for model_list
results <- resamples(model_list)
xyplot(results)
summary(results)
# correaltion among the models
modelCor(results)
#       glm          lda           rf          knn
# glm 1.0000000000 0.6928015987 0.7797776637 0.7633080697
# lda 0.6928015987 1.0000000000 0.3568622263 0.6550666742
# rf  0.7797776637 0.3568622263 1.0000000000 0.4972263798
# knn 0.7633080697 0.6550666742 0.4972263798 1.0000000000

# high correlation between : glm and rf = 0.7797776637, glm and knn =0.7633080697, 
splom(results)
# Highly correlated model, try with other algos in model_list_1
model_preds <- lapply(model_list, predict, newdata=tdata[, features_flag], type="prob")

###############################################################################################
# check correlation and results for model_list_1

results_2 <- resamples(model_list_1)
xyplot(results_2)
summary(results_2)  # rf creates the best model with mean accuracy 0.9532576898

# correaltion among the models
modelCor(results_2)
#           rpart          lda           rf           knn
# rpart 1.00000000000 0.5945074165 0.6872268804 0.08701838999
# lda   0.59450741651 1.0000000000 0.5619465779 0.54325792012
# rf    0.68722688044 0.5619465779 1.0000000000 0.27295698500
# knn   0.08701838999 0.5432579201 0.2729569850 1.00000000000
# all correlation values are less than 0.75 so we can consider all the models for stacking 

splom(results_2)

# create a stacked model using a simple linear model using caretStack()
## stack using glm

set.seed(20)
glm_ensemble <- caretStack(model_list_1, method ="glm", trControl = trainControl(method= "repeatedcv", number =5,
                                                                                 savePredictions ="final", classProbs=TRUE ))
print(glm_ensemble)  # Accuracy lifted to 0.9568235281
predict_glm_ensemble <- predict(glm_ensemble, newdata = tdata[, features_flag]) 
predict_glm_ensemble.prob <- predict(glm_ensemble, newdata = tdata[, features_flag], type='prob') 
cm = confusionMatrix(predict_glm_ensemble, test_class_st) 

#             Reference
#Prediction   neg     pos
#         neg 14832     9
#         pos   793   366

total_cost = 793*10+ 500* 9 #12430

# #  Stacking - using original data

# Splitting training data into 40/60  based on the dependent variable
set.seed(20)
sampling_data_s <- sampling_data
sampling_data_s$num_class <- ifelse(sampling_data_s$num_class==1, "pos","neg")

index = createDataPartition(sampling_data_s$num_class, p=0.40, list=FALSE)  
traindata = sampling_data_s[index,]
validdata = sampling_data_s[-index,]

# create submodels
set.seed(20)
model_list_2 <- caretList(num_class ~ . , data= traindata,trControl = fitControl, methodList =c("rpart", "lda", "rf","knn"))

# check correlation and results for model_list
results_train <- resamples(model_list_2)
xyplot(results_train)
summary(results_train)
#Call:
# summary.resamples(object = results_train)

#Models: rpart, lda, rf, knn 
#Number of resamples: 15 

#Accuracy 
#           Min.      1st Qu.       Median         Mean      3rd Qu.         Max. NA's
# rpart 0.9866666667 0.9875000000 0.9879166667 0.9878611111 0.9883333333 0.9887500000    0
# lda   0.9760416667 0.9778125000 0.9787500000 0.9788194444 0.9793750000 0.9827083333    0
# rf    0.9891666667 0.9908333333 0.9912500000 0.9910972222 0.9916666667 0.9922916667    0
# knn   0.9833333333 0.9855208333 0.9864583333 0.9861111111 0.9868750000 0.9877083333    0

#Kappa 
#               Min.      1st Qu.       Median         Mean      3rd Qu.         Max. NA's
#rpart 0.4465592972 0.5180795759 0.5461422088 0.5363769605 0.5664059124 0.6042216359    0
#lda   0.4486175483 0.5046931058 0.5158398087 0.5181254462 0.5323615536 0.5906625021    0
#rf    0.6066565809 0.6699882431 0.6875793752 0.6866108080 0.7199404980 0.7299927025    0
#knn   0.2788461538 0.3646567524 0.3866973798 0.3891953127 0.4320482200 0.4816983895    0

# correlation among the models
modelCor(results_train)
#         rpart           lda             rf            knn
#rpart 1.00000000000 0.24247257817  0.06850923553  0.50901369063
#lda   0.24247257817 1.00000000000  0.10563817053  0.04228657263
#rf    0.06850923553 0.10563817053  1.00000000000 -0.06067503882
#knn   0.50901369063 0.04228657263 -0.06067503882  1.00000000000

# no correlation between any model 
splom(results_train)

# predict on validation data
model_preds_tr <- lapply(model_list_2, predict, newdata=validdata[, -1], type="prob")

#create a stacked model using a simple linear model using caretStack()
# # stack using glm
set.seed(20)
glm_ensemble_s <- caretStack(model_list_2, method ="glm", trControl = trainControl(method= "repeatedcv", number =5,
                                                                                   savePredictions ="final", classProbs=TRUE ))
print(glm_ensemble_s)  # Accuracy lifted to 0.9910138889 : it not better than the one we got with rf 

predict_glm_ensemble_s <- predict(glm_ensemble_s, newdata = tdata[, features_flag]) 
predict_glm_ensemble_s.prob <- predict(glm_ensemble_s, newdata = tdata[, features_flag], type='prob') 
cm_s = confusionMatrix(predict_glm_ensemble_s, test_class_st) 

#             Reference
#Prediction   neg     pos
#         neg 15590   128
#         pos   35   247
total_cost <- sum(10*((predict_glm_ensemble_s=='pos')&(test_class_st=='neg'))+500*((predict_glm_ensemble_s=='neg')&(test_class_st=='pos')))
#64350  (total_cost = 35*10+ 500* 128)


#############################################################################################
# Features importance calculation

# Reading pre-processed data into R
data <- read.csv(file="use_me_train_data.csv", header=TRUE)
tdata <- read.csv(file="use_me_test_data.csv", header=TRUE)
num_class <- ifelse(data$class=="pos",1,0)
test_class <- ifelse(tdata$class=="pos",1,0)

hist_labels <- c("ag_000", "ag_001", "ag_002", "ag_003", "ag_004", "ag_005", "ag_006", "ag_007", "ag_008", "ag_009", "ay_000", "ay_001", "ay_002", "ay_003", "ay_004", "ay_005", "ay_006", "ay_007", "ay_008", "ay_009", "az_000", "az_001", "az_002", "az_003", "az_004", "az_005", "az_006", "az_007", "az_008", "az_009", "ba_000", "ba_001", "ba_002", "ba_003", "ba_004", "ba_005", "ba_006", "ba_007", "ba_008", "ba_009", "cn_000", "cn_001", "cn_002", "cn_003", "cn_004", "cn_005", "cn_006", "cn_007", "cn_008", "cn_009", "cs_000", "cs_001", "cs_002", "cs_003", "cs_004", "cs_005", "cs_006", "cs_007", "cs_008", "cs_009", "ee_000", "ee_001", "ee_002", "ee_003", "ee_004", "ee_005", "ee_006", "ee_007", "ee_008", "ee_009")
features_flag <- rep(TRUE, ncol(data))
for(j in 1:length(hist_labels)){
  i <- which(colnames(data)==hist_labels[j])
  features_flag[i] <- FALSE
}

features_flag[1:2] <- FALSE
sampling_data <- cbind(num_class,data[,features_flag])
library(ROSE)
# oversampling of the minority class
ovdata <- ovun.sample(num_class~., data=sampling_data, method="over", p=0.5, seed=1)$data
undata <- ovun.sample(num_class~., data=sampling_data, method="under", p=0.5, seed=1)$data

library(caret)

fitControl <- trainControl(method="repeatedcv", number=5, repeats=5, returnResamp="all", classProbs=TRUE)
grid <- expand.grid(.winnow=c(TRUE,FALSE), .trials=c(1,5,10,15,20), .model="tree")

x <- undata[,-1]
y <- ifelse(undata[,1]==1, 'pos', 'neg')
under_c50 <- train(x=x, y=y, tuneGrid=grid, trControl=fitControl, method="C5.0", verbose=FALSE)
roc_imp <- varImp(under_c50)
roc_imp

# C5.0 variable importance
#       Overall
# cl_000  100.00
# am_0    100.00
# aa_000  100.00
# cn_m    100.00
# ee_m    100.00
# ag_m    100.00
# ck_000  100.00
# by_000  100.00
# ar_000  100.00
# ai_000   99.80
# az_m     98.67
# cg_000   97.76
# bu_000   97.25
# bm_000   96.79
# do_000   96.02
# bk_000   94.44
# cj_000   92.91
# aj_000   88.78
# dg_000   88.63
# dp_000   87.97

sampling_data[,1] <- ifelse(sampling_data[,1]==1, "pos", "neg")
original_LR <- train(num_class~., data=sampling_data, method='glm', family='binomial', metric="Kappa", trControl=fitControl)
lr_imp <- varImp(original_LR)
lr_imp
# glm variable importance
#        Overall
# aa_000  100.00
# bi_000   97.43
# bs_000   96.41
# cb_000   94.34
# bu_000   85.91
# cs_m     85.51
# ay_m     70.26
# ee_f     68.57
# ai_000   63.74
# cn_m     44.83
# am_0     40.63
# at_000   40.06
# cs_f     36.58
# av_000   36.35
# dr_000   35.40
# by_000   34.14
# bd_000   33.47
# cl_000   32.30
# az_m     31.13
# db_000   30.13

under_LDA <- train(y~., data=cbind(y,x), method='lda', metric='Kappa', trControl=fitControl)
lda_imp <- varImp(under_LDA)
lda_imp
# ROC curve variable importance
#        Importance
# ck_000     100.00
# aa_000      98.59
# bu_000      93.52
# bi_000      91.89
# cn_m        87.62
# by_000      86.35
# ba_m        86.17
# ag_m        85.46
# dd_000      85.14
# az_m        83.61
# am_0        82.39
# de_000      80.04
# bd_000      74.05
# be_000      73.63
# ee_m        70.96
# eb_000      70.44
# cs_m        69.84
# bc_000      69.44
# ai_000      69.13
# bz_000      65.97

under_svm <- train(x=x, y=y, method="svmLinear", preProc=c("center","scale"), metric="Kappa", trControl=fitControl)
svm_imp <- varImp(under_svm)
svm_imp
# ROC curve variable importance
#       Importance
# ck_000     100.00
# aa_000      98.59
# bu_000      93.52
# bi_000      91.89
# cn_m        87.62
# by_000      86.35
# ba_m        86.17
# ag_m        85.46
# dd_000      85.14
# az_m        83.61
# am_0        82.39
# de_000      80.04
# bd_000      74.05
# be_000      73.63
# ee_m        70.96
# eb_000      70.44
# cs_m        69.84
# bc_000      69.44
# ai_000      69.13
# bz_000      65.97

under_rf <- train(y~., cbind(y,x), method="rf", metric="Kappa", trControl=fitControl)
rf_imp <- varImp(under_rf)
rf_imp
# rf variable importance
#        Overall
# ck_000 100.0000
# aa_000  67.0054
# bu_000  29.4690
# cn_m    12.8696
# bi_000  8.6321
# ag_m     4.1944
# ba_m     3.4085
# am_0     3.0166
# by_000   2.9899
# bk_000   1.9513
# az_m     1.5177
# bl_000   1.5041
# be_000   1.3139
# ac_000   1.2218
# cs_m     1.0486
# bn_000   0.9860
# dt_000   0.9221
# bm_000   0.9221
# ai_000   0.9031
# dx_000   0.8829
