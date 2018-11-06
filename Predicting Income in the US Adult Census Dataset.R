library(ggplot2)
library(ggthemes)
library(dplyr)
library(mice)
library(VIM)
library(tree)
library(e1071)
library(caret)
library(C50)
library(GoodmanKruskal)
#install.packages("caret")
setwd("/Users/gawel/Desktop/ma429Mock")

# downloading data
con <- file("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", "r")
data <- read.csv(con, header=F, sep=",", strip.white=T, na.strings="?")
close(con)
write.table(data, "adult.data", sep=",")

con <- file("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", "r")
testdata <- read.csv(con, header=F, skip=1, sep=",", strip.white=T, na.strings="?")
close(con)
write.table(testdata, "adult.test", sep=",")

# start working with the data
data = read.table("adult.data" , sep=",")
testdata = read.table("adult.test", sep=",")
colnames(data) <- c("age","workclass","fnlwgt","education","edunum","maritalstatus","occupation"
                 ,"relationship","race","sex","capitalgain","capitalloss","hpw","country","income")
colnames(testdata) <- colnames(data)
summary(data)
summary(testdata)

# whole database statistics

testTemp <- testdata
testTemp$income = ifelse(testTemp$income == ">50K.",">50K","<=50K")
database <- rbind(data,testTemp)
colnames(database) <- colnames(data)
database$income = as.factor(database$income)
summary(database)
md.pattern(database)
aggr_plot <- aggr(database, numbers=TRUE, sortVars=TRUE, labels=names(database))
rm(database, testTemp)
# fix(data)
# fix(testdata)

# checking for NA values
sum(is.na(data))
dim(data)
md.pattern(data)
aggr_plot <- aggr(data, numbers=TRUE, sortVars=TRUE, labels=names(data))

# drop NA values, consider complete sets only
# fdata = data[complete.cases(data),]
# ftestdata = testdata[complete.cases(testdata),]
# sum(is.na(fdata))
# sum(is.na(ftestdata))

#####################################################################
# Imputing the missing values in training and test datasets
# We need them as all cases have to be predicted, ideally

tempdata <- mice(data, m=1, maxit=10, meth='pmm', seed=1)
fdata <- complete(tempdata,1) # imputed only 1 dataset, so no benefits from pooling results possible

temptestdata <- mice(testdata, m=1, maxit=10, meth='pmm', seed=1)
ftestdata <- complete(temptestdata,1) # same - one dataset imputed
#####################################################################

# Adding a new variable for income: having two levels 0 and 1
fdata$income1 = ifelse(fdata$income == ">50K",1,0)

ftestdata$income = ifelse(ftestdata$income == ">50K.",">50K","<=50K")
ftestdata$income1 = ifelse(ftestdata$income == ">50K",1,0)
ftestdata$income = as.factor(ftestdata$income)

head(fdata)
head(ftestdata)

####################################################
# Visualisation
attach(fdata)
a_new = subset(fdata, income == ">50K")
b_new = subset(fdata, income == "<=50K")

# age vs income distribution
ggplot(a_new, aes(age, ..count..,fill=income)) + geom_histogram(binwidth=3,col="black") 
ggplot(b_new, aes(age, fill=factor(income))) + geom_histogram(binwidth=3,col="black") 
ggplot(fdata, aes(age, fill=factor(income))) + geom_histogram(binwidth=5) 

# education vs income
#ggplot(fdata, aes(x=education, fill = income))+geom_bar(position="dodge")
ggplot(fdata, aes(x=education, fill = income))+geom_bar(position="fill")
ggplot(fdata, aes(x=edunum, fill = income))+geom_bar(position="fill")

# workclass vs income,
ggplot(fdata, aes(x=workclass, fill = income))+geom_bar(width= 0.2)
plot( fdata$workclass,fdata$income,xlab = "workclass", ylab ="income")

# hpw vs income
ggplot(fdata, aes(x = hpw, fill = income))+geom_bar(position="fill")

# country vs income
countryd = fdata %>% group_by(country) %>% filter(n() > 100)
table(countryd$country,countryd$income)
ggplot(countryd, aes(x=country, fill = income))+geom_bar(width= 0.1,position="fill")
plot(countryd$country,countryd$income)



# train and test data using the adult.data
set.seed(2)
#train = sample(1:nrow(fdata), size= 20000) # 75% of total data
traindata = fdata[,-c(15,16)]
ytrain = fdata[,16]
head(traindata)
dim(traindata)

test = ftestdata[,-c(15,16)]
ytest = ftestdata[,15]
dim(test)
# str(fdata)

####################################################
# Naive Bayes classification

library(naivebayes)

nb <- naive_bayes(x=fdata[,-c(15,16)], y=fdata[,16])
pred <- predict(nb, newdata=ftestdata[,-16],type="prob")

table((pred[,2]>=0.5), ftestdata[,16])
mean((pred[,2]>=0.5) == (ftestdata[,16]==">50K"))	
mean((pred[,2]>=0.5) != (ftestdata[,16]==">50K"))

# for imputed dataset
#pred\ytest     FALSE  TRUE
#  	FALSE 11766  2282
#	TRUE    669  1564

# MER (Misclassification error rate)
(669+2282)/(669+2282+11766+1564)
# 18.1% missclassification error

#Precision rate
(1564)/(1564+669)
# 70,0%

#Recall rate
(1564)/(1564+2282)
# 40,7%

####################################################
# Logistic Regression model
# Training the model with full adult.data dataset

fulltraindata = fdata[,-c(15,16)]
ytrain1 = fdata[,16]
length(ytrain1)

fulltestdata = ftestdata[,-c(15,16)]
ytest1 = ftestdata[,15]
length(ytest1)

# Logistic model
glm.mod = glm(ytrain1 ~ . , data = fulltraindata, family = binomial)
# predictions on the adult.test dataset
pred.probfull = predict(glm.mod, newdata = fulltestdata, type = "response")
fullprobs = rep("<=50K",nrow(fulltestdata))
fullprobs[pred.probfull  > 0.5]=">50K"
table(fullprobs, ytest1)

# for imputed dataset
#         ytest1
#fullprobs <=50K  >50K
#    <=50K 11580  1572
#    >50K    855  2274

# MER (Misclassification error rate)
mean(fullprobs!=ytest1)

# precison 
2274/(2274+855)   #72.67
11580/(11580+1572) #88
# recall rate
2274/(2274+1572)

# Step wise   --- does not improve anything if we drop fnlwgt and edunum

glm.mods = glm(ytrain1 ~ . -fnlwgt - education   , data = fulltraindata, family = binomial)

pred.probfull1 = predict(glm.mods, newdata = fulltestdata, type = "response")
fullprobs1 = rep("<=50K",nrow(fulltestdata))
fullprobs1[pred.probfull1  > 0.5]=">50K"
table(fullprobs1, ytest1)
mean(fullprobs1!=ytest1)   #15.27
#fullprobs1 <=50K  >50K
#<=50K 10522  1462
#>50K    838  2238
# precison
2238/(838+2238)   # 72.75
10522/(10522+1462) #87

# backward stepwise : this also drops fnlwgt and edunum only.
bw = step(glm.mod)

####################################################
# Trees classification on full training dataset
tree_data = fdata[,-c(16)]
tree.mod = tree(income ~ . - country, tree_data)
summary(tree.mod)   
plot(tree.mod)
text(tree.mod)	
tree.mod

# Test data
tree_testdata = ftestdata[,-c(15,16)]
y = ftestdata[,15]
pred = predict(tree.mod, tree_testdata,type="class")
table(pred, y)
mean(pred==y)
mean(pred!=y)

# for imputed dataset
#       y
# pred    <=50K  >50K
#  <=50K 11805  1901
#  >50K    630  1945

# MER (missclassification error rate)
mean(pred!=y)

# Precision rate
1945/(1945+630)

# Recall rate
1945/(1945+1901)


# Tree pruning
set.seed(3)
cv.treemod = cv.tree(tree.mod,FUN=prune.misclass)
cv.treemod  

old <- par()
par(mfrow=c(1,2))
plot(cv.treemod$size, cv.treemod$dev, type="b")
plot(cv.treemod$k, cv.treemod$dev, type="b")

# tree with 5 terminal nodes has the lowest CV error
prune.treemod <- prune.misclass(tree.mod, best=5)
prune.pred <- predict(prune.treemod, tree_testdata,type="class")
table(pred,y)
mean(pred==y)	
mean(pred!=y)

# for imputed dataset
#       y
#pred    <=50K  >50K
#  <=50K 11805  1901
#  >50K    630  1945
# 15.5% missclassification error

####################################################
# Support Vector Machine

set.seed(1)
train <- sample(1:dim(fdata)[1],2000) 
xtrain <- fdata[,-c(15,16)]
ytrain <- fdata[,15]

# MIND THE RUNNING TIME - use subset
 set.seed(1)
 tune.out <- tune(svm, ytrain~., data=cbind(xtrain,ytrain), kernel="linear", ranges=list(cost=c(0.01,0.1,1,10)))
 summary(tune.out)
 svm.model <- tune.out$best.model
 summary(svm.model)
#- Detailed performance results:
#   cost  error dispersion
#1  0.01 0.1675 0.01975545
#2  0.10 0.1505 0.02650472
#3  1.00 0.1460 0.02633122
#4 10.00 0.1525 0.02474874

# Train and test on FULL data sets
xtrain <- fdata[,-c(15,16)]
ytrain <- fdata[,15]

# MIND THE RUNNING TIME - about 4 mins
svm.model = svm(ytrain~age+workclass+fnlwgt+education+edunum+maritalstatus+occupation+relationship+race+sex+capitalgain+capitalloss+hpw+country, data=cbind(xtrain,ytrain), kernel ="linear", cost=1, scale = T)
summary(svm.model)

# watch that data definition, classes, and factor levels are the same for training and test data
testd <- ftestdata[,-c(16)]
for (i in 1:dim(xtrain)[2]){
	class(testd[,i]) <- class(xtrain[,i])
	if (is.factor(testd[,i])) {
		levels(testd[,i]) <- levels(xtrain[,i])
		attr(testd[,i],"contrasts") <- attr(xtrain[,i],"contrasts")

	}
}
str(xtrain)
str(testd)

ypred <- predict(svm.model, newdata=testd[,-15])
table(ypred, testd[,15])
mean(ypred==testd[,15])
mean(ypred!=testd[,15])


# for imputed dataset
# ypred   <=50K  >50K
#  <=50K 11816  1833
#  >50K    619  2013

# MER (Misclassification error rate)
mean(ypred!=testd[,15])

# Precison
2013/(2013+619)

# Recall
2013/(2013+1833)



####################################################
# C5.0 tree algorithm

x <- fdata[,-c(15,16)]
y <- fdata[,15]
fitControl <- trainControl(method="repeatedcv",number=10,repeats=10,returnResamp="all")
grid <- expand.grid(.winnow=c(TRUE,FALSE), .trials=c(1,5,10,15,20), .model="tree")

# MIND THE RUNNING TIME - 30mins
c50.tree <- train(x=x,y=y,tuneGrid=grid,trControl=fitControl,method="C5.0", verbose=FALSE)
plot(c50.tree)

pred <- predict(c50.tree, ftestdata[,-c(15,16)])
table(pred,ftestdata[,15])
mean(pred==ftestdata[,15])


# for imputed dataset
#pred    <=50K  >50K
#  <=50K 11648  1346
#  >50K    787  2500
# 13.1% missclassification error

# MER (Misclassfication error rate)
mean(pred!=ftestdata[,15])

# Precision rate
2500/(2500+787)

# Recall rate
2500/(2500+1346)


# Correlation 
str(fulltraindata)
cormat = cor(fulltraindata[,c("age","edunum", "capitalgain","capitalloss","hpw","fnlwgt")])
print(cormat)
highcor= findCorrelation(cormat, cutoff=0.2)
print(highcor) # no correlation among the numerical varables


str(traindata)
catvar = c("workclass","education","maritalstatus","occupation","relationship","race","sex","country")
catdata = subset(traindata, select = catvar)
GKmatrix1 = GKtauDataframe(catdata)
plot(GKmatrix1)
# Coorelation among the categorical variables only - marital status is correlated with sex :cor value=0.22, sex not so much correlated with marital status, value=0.1 
# maritalstatus with relationship, value = 0.42, relationship with maritalstatus, value=0.59

catvar1 = c("workclass","education","maritalstatus","occupation","relationship","race","sex","country","edunum","age")
catdata1 = subset(traindata, select = catvar1)
GKmatrix2 = GKtauDataframe(catdata1)
plot(GKmatrix2)
# all variables of the training data with first 10000 observations
GKmatrix3 = GKtauDataframe(traindata[1:10000,-c(3,14)] )
plot(GKmatrix3)

#  Random Forest
#tree_data = fdata[,-c(16)]
library(randomForest)
bag.fdata = randomForest(income ~ . , data = tree_data, mtry = 14, importance= T)
bag.fdata
test = ftestdata[,-c(16)]
levels(test) = levels(tree_data)
bag.pred= predict(bag.fdata, newdata =test[,-15])


