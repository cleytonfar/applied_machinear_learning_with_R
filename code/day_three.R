library(caret)

## In the following examples, we will use the boosting algorithm to run different
## versions using different performance metric.
## Note that we will be used the 10-fold cross-validation technique. 

## Using Kappa as performance metric:
model.fit <- train(Species ~ .,
                   data = iris,
                   method = "gbm",
                   metric = "Kappa",
                   trControl = trainControl(method = "cv", 
                                            number = 10, 
                                            classProbs = T,
                                            summaryFunction = multiClassSummary))

## Using the LogLoss as performance metric:
model.fit <- train(Species ~ .,
                   data = iris,
                   method = "gbm",
                   metric = "logLoss",
                   trControl = trainControl(method = "cv", number = 10, 
                                            classProbs = T,
                                            summaryFunction = multiClassSummary))


## Using ROC-AUC:
## REMEMBER: ROC metrics are only suitable for binary classification problems!!!
library(mlbench) ## to load the PimaIndiansDiabetes
data("PimaIndiansDiabetes")
model.fit <- train(diabetes ~.,
                   data = PimaIndiansDiabetes,
                   method = "gbm",
                   metric = "ROC",
                   trControl = trainControl(method = "cv", 
                                            number = 10, 
                                            classProbs = T, 
                                            summaryFunction = twoClassSummary))

## YOUR TURN:
## Try to the same exercises using different algorithms and try to tweak some of 
## the arguments in the train() function from caret.
## OBS: use the help page to see information about train() function.

