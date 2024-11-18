domains <- read.csv("C://Users/USER/Desktop/IC SE/3-1/Machine Learning/ML project/legit-dga_domains.csv",stringsAsFactors = FALSE, fileEncoding="UTF-8-BOM")
domains <- domains[,c(2,3)]
domains <- domains[1:10000,]
#domains$class <- factor(domains$class)
domains$subclass <- factor(domains$subclass)

train_amount <- sample.int(nrow(domains), size = nrow(domains)*0.7)
domains_train <- domains[train_amount,]
domains_test <- domains[-train_amount,]

#install.packages("e1071")
library("e1071")
#install.packages("RTextTools")
library(RTextTools)
library(tm)

# Create the document term matrix. If column name is domain
dtMatrix <- create_matrix(domains["domain"])

#dtMatrix <- removeSparseTerms(dtMatrix, sparse = .999)

# Configure the training data
container <- create_container(dtMatrix, domains$subclass, trainSize=train_amount, virgin=FALSE)
model <- train_model(container, "SVM", kernel="radial", cost=1, gamma=0.5)

data_test <- domains_test["domain"]
predictionData <- split(data_test$host, seq(nrow(data_test)))
#predictionData <- as.list(data_test)
predMatrix <- create_matrix(predictionData, originalMatrix = dtMatrix)
# create the corresponding container
predSize = length(predictionData);
predictionContainer <- create_container(predMatrix, labels=rep(0,predSize), testSize=1:predSize, virgin=FALSE)

results <- classify_model(predictionContainer, model)
results
library(gmodels)
CrossTable(domains_test$subclass, results$SVM_LABEL, prop.chisq = FALSE, 
           prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))
#plot(results)


#Working SVM
domains <- read.csv("C://Users/USER/Desktop/IC SE/3-1/Machine Learning/ML project/legit-dga_domains.csv",stringsAsFactors = FALSE, fileEncoding="UTF-8-BOM")
domains <- domains[,c(2,4)]
#domains <- domains[109001:119000,]
domains$subclass <- factor(domains$subclass)

summary(domains)

#Features
#a) domain name length
domains_length <- data.frame(length=apply(domains,2,nchar)[,1])

#b) domain name entropy
#install.packages("acss")
#package for get string entropy
library(acss)
domains_entropy <- data.frame(entropy=entropy(domains[,1]))


#c) vowel's ratio in domain name
vowels <- function(x) {
  nchar(gsub("[^aeiou]","",x, ignore.case = TRUE))
}
vowel_ratio <- function(x) {
  nchar(gsub("[^aeiou]","",x, ignore.case = TRUE))/nchar(x)
}
domains_vowel <- data.frame(vowel_ratio=vowel_ratio(domains[,1]))

#d) consectutive consonants' ratio
domains_consecutive_constants <- lapply(gregexpr("[^aeiou0-9][^aeiou0-9]+",domains[,1],ignore.case = TRUE),function(x) length(x[x > 0]))
domains_consecutive_ratio <- data.frame(consecutive_ratio=unlist(domains_consecutive_constants))
domains_consecutive_ratio <- domains_consecutive_ratio/domains_length

#e) proportion of the digits
digit_ratio <- function(x) {
  nchar(gsub("[^0-9]","",x, ignore.case = TRUE))/nchar(x)
}
domains_digit <- data.frame(digit_ratio=digit_ratio(domains[,1]))

# Merge table
data <- cbind(domains_length,domains_entropy,domains_vowel,domains_consecutive_ratio,domains_digit,domains["subclass"])

#separate training and test set
train_amount <- sample.int(nrow(data), size = nrow(data)*0.7)
domains_train <- data[train_amount,]
domains_test <- data[-train_amount,]

library(e1071)
model <- svm(subclass ~ .,data = domains_train, kernel = "linear", probability = TRUE)
domains_pred <- predict(model,domains_test)

#Visualization
#CrossTable
#library(gmodels)
#CrossTable(domains_pred,domains_test$subclass, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, dnn = c('predicted', 'actual'))

#Scatterplot
#pairs(subclass ~ ., data = domains_train, col = domains_train$subclass)

#Confusion Matrix
library(caret)
confusionMatrix(domains_pred, domains_test$subclass)

#Radial SVM
model2 <- svm(subclass ~ .,data = domains_train, kernel = "radial", cost = 1, gamma = 0.5, probability = TRUE)
domains_pred2 <- predict(model2,domains_test)

#Confusion Matrix
library(caret)
confusionMatrix(domains_pred2, domains_test$subclass)

#Tuning using grid search and cross validation
svmTune <- tune(svm, subclass ~ ., data = domains_train, ranges = list(cost = 10^(0:4), gamma = 10^(-6:3)))
#summary(svmTune) #cost = 100, gamma = 0.1

#Radial SVM after Cost-Validataion
model3 <- svm(subclass ~ .,data = domains_train, kernel = "radial", cost = 100, gamma = .01, probability = TRUE)
domains_pred3 <- predict(model3,domains_test)

confusionMatrix(domains_pred3, domains_test$subclass)

#scale the data
domains_scale_train <- as.data.frame(sapply(domains_train[,-6], function(x) if(is.numeric(x)) scale(x) else x))
domains_scale_train <- cbind(domains_scale_train,subclass = domains_train[,6])
domains_scale_test <- as.data.frame(sapply(domains_test[,-6], function(x) if(is.numeric(x)) scale(x) else x))
domains_scale_test <- cbind(domains_scale_test,subclass = domains_test[,6])

#SVM Linear
model_scale <- svm(subclass ~ .,data = domains_scale_train, kernel = "linear", probability = TRUE)
domains_scale_pred <- predict(model_scale,domains_scale_test)

confusionMatrix(domains_scale_pred,domains_scale_test$subclass)

#SVM Radial (default)
model2_scale <- svm(subclass ~ .,data = domains_scale_train, kernel = "radial", cost = 1, gamma = 0.5, probability = TRUE)
domains_scale_pred2 <- predict(model2_scale,domains_scale_test)

confusionMatrix(domains_scale_pred2,domains_scale_test$subclass)

#Cross Validation scale data
svmTune2 <- tune(svm, subclass ~ ., data = domains_scale_train, ranges = list(cost = 2^seq(-3,11,by = 2), gamma = 2^seq(-7,3,by = 2)))

#SVM Radial (Cost = 8, gamma = 0.5)
model3_scale <- svm(subclass ~ .,data = domains_scale_train, kernel = "radial", cost = 8, gamma = 0.5, probability = TRUE)
domains_scale_pred3 <- predict(model3_scale,domains_scale_test)

confusionMatrix(domains_scale_pred3,domains_scale_test$subclass)
