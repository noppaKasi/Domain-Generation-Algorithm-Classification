library(e1071) #package for SVM
library(acss) #package for get string entropy
library(caret) #for visualizing Confusion Matrix

#Import and Preparing data
domains <- read.csv("C://Users/USER/Desktop/IC SE/3-1/Machine Learning/ML project/legit-dga_domains.csv",stringsAsFactors = FALSE, fileEncoding="UTF-8-BOM")
#Use only domain and subclass
domains <- domains[,c(2,4)]
#Convert subclass into factor
#1 = cryptolocker, 2 = dga, 3 = legit, 4 = newdga
domains$subclass <- factor(domains$subclass)

#Features
#a) domain name length
domains_length <- data.frame(length=apply(domains,2,nchar)[,1])

#b) domain name entropy
domains_entropy <- data.frame(entropy=entropy(domains[,1]))


#c) vowel's ratio in domain name
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

# Merge all features into one data frame
data <- cbind(domains_length,domains_entropy,domains_vowel,domains_consecutive_ratio,domains_digit,domains["subclass"])

#separate training and test set
train_amount <- sample.int(nrow(data), size = nrow(data)*0.7)
domains_train <- data[train_amount,]
domains_test <- data[-train_amount,]

#Visualization data with Scatterplot
pairs(subclass ~ ., data = domains_train, col = domains_train$subclass)

#Linear SVM
model <- svm(subclass ~ .,data = domains_train, kernel = "linear", probability = TRUE)
domains_pred <- predict(model,domains_test)

confusionMatrix(domains_pred, domains_test$subclass)

#Radial SVM
model2 <- svm(subclass ~ .,data = domains_train, kernel = "radial", cost = 1, gamma = 0.5, probability = TRUE)
domains_pred2 <- predict(model2,domains_test)

confusionMatrix(domains_pred2, domains_test$subclass)

#Tuning using grid search and cross validation
svmTune <- tune(svm, subclass ~ ., data = domains_train, ranges = list(cost = 10^(0:4), gamma = 10^(-6:3)))
summary(svmTune) #cost = 100, gamma = 0.1

#Radial SVM after Cost-Validataion
model3 <- svm(subclass ~ .,data = domains_train, kernel = "radial", cost = 100, gamma = .01, probability = TRUE)
domains_pred3 <- predict(model3,domains_test)

confusionMatrix(domains_pred3, domains_test$subclass)

#scale the data
domains_scale_train <- as.data.frame(sapply(domains_train[,-6], function(x) if(is.numeric(x)) scale(x) else x))
domains_scale_train <- cbind(domains_scale_train,subclass = domains_train[,6])
domains_scale_test <- as.data.frame(sapply(domains_test[,-6], function(x) if(is.numeric(x)) scale(x) else x))
domains_scale_test <- cbind(domains_scale_test,subclass = domains_test[,6])

#Scaled Linear SVM
model_scale <- svm(subclass ~ .,data = domains_scale_train, kernel = "linear", probability = TRUE)
domains_scale_pred <- predict(model_scale,domains_scale_test)

confusionMatrix(domains_scale_pred,domains_scale_test$subclass)

#Scaled Radial SVM (default)
model2_scale <- svm(subclass ~ .,data = domains_scale_train, kernel = "radial", cost = 1, gamma = 0.5, probability = TRUE)
domains_scale_pred2 <- predict(model2_scale,domains_scale_test)

confusionMatrix(domains_scale_pred2,domains_scale_test$subclass)

#Cross Validation scale data
svmTune2 <- tune(svm, subclass ~ ., data = domains_scale_train, ranges = list(cost = 2^seq(-3,11,by = 2), gamma = 2^seq(-7,3,by = 2)))
summary(svmTune2)

#Scaled Radial SVM (Cost = 8, gamma = 0.5)
model3_scale <- svm(subclass ~ .,data = domains_scale_train, kernel = "radial", cost = 8, gamma = 0.5, probability = TRUE)
domains_scale_pred3 <- predict(model3_scale,domains_scale_test)

confusionMatrix(domains_scale_pred3,domains_scale_test$subclass)