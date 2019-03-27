library(data.table)
library(ggplot2)
library(ISLR)
library(tidyr)
library(dplyr)

## The Stock Market Data:
stock <- data.table(Smarket)
help("Smarket")

## This data consists of percentage returns for the S&P 500 stock over 1,250 days.
##  - Lag1 - Lag5: percentage return for each of the five previous trading days;
##  - Volume: number of shares traded on the previous day
##  - Today: the percentage return on the date in question
##  - Direction: whether the market was Up or Down on this date

## Correlation:
cor_stock <- cor(stock)
stock %>% str

## Converting the Direction variable to numeric:
cor_stock <- stock %>% 
    mutate(Direction = as.numeric(Direction) - 1) %>% 
    cor


cor_stock

ggplot(stock) +
    aes(x = 1:nrow(stock), y = Volume) + 
    geom_line() + 
    theme_minimal() + 
    labs(x = "", y = "volume")


## Let's fit a model to predict Direction using Lag1-Lag5 and Volume:

## Let's check the proportion of each class:
stock %>% 
    pull(Direction) %>% 
    table() %>% 
    prop.table()

##----------------------------------------------------------------------------##
## First things first

stock %>% pull(Direction) %>% levels()
stock$Direction
## Let's make "Up" the firt level
stock <- stock %>% 
    mutate(Direction = relevel(Direction, c("Up"))) %>% 
    as.data.table()
stock$Direction
stock %>% pull(Direction) %>% levels()

set.seed(27)
inTrain <- sample(nrow(stock), size = 0.8*nrow(stock))
inTrain

##----------------------------------------------------------------------------##
## First things first

## Let's separate the training and validation set:
## Since the dataset is order through time, let's take the obs until 2004 as a
## training set and using the obs from 2005 as a validation set:
training <- stock[Year <= 2004]
validation <- stock[Year > 2004]


## Let's fit a model to predict Diraction using Lag1-Lag5 and Volume:
## Our strategy: fit the algorithm on training set and test on validation set


##----------------------------------------------------------------------------##
## Logistic model:

## In order to run a logistic algorithm, we use the glm() function with the 
## argument family = "binomial"
logit <- glm(formula = Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume,
             family = binomial(link = "logit"), 
             data = training)

summary(logit)
logit.pred <- predict(logit, validation)
logit.pred

## 
mean(validation$Direction == ifelse(logit.pred > 0.5, "Up", "Down"))


##----------------------------------------------------------------------------##
## Linear Discriminant Analysis:

## We perform LDA using the lda() function from the MASS library.
library(MASS)
lda.fit <- lda(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume, 
               data = training)

lda.fit

lda.pred <- predict(lda.fit, validation)
lda.pred %>% length
lda.pred %>% class

names(lda.pred)
lda.pred$class
lda.pred$posterior
## Performance:
mean(validation$Direction == lda.pred$class)



## Quadratic Discriminant Analysis:

## QDA is implemented in R using the qda() function, also from MASS library.

qda.fit <- qda(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume, 
               data = training)

qda.fit

qda.pred <- predict(qda.fit, validation)

## Performance:
mean(validation$Direction == qda.pred$class)


## Comparing the algorithms:
mean(validation$Direction == ifelse(logit.pred > 0.5, "Up", "Down"))
mean(validation$Direction == lda.pred$class)
mean(validation$Direction == qda.pred$class)



## K-Nearests Neighbors:

## To perform KNN we use the knn() function from the class library.
library(class)

## Rather than a two-step approach in which we first fit the model and then use
## the model to make predictions, knn() forms predictions using a single command.
## The function requires 4 inputs:
##  1. matrix containing the predictors from the training data;
##  2. matrix containing the predictors from the test data;
##  3. vector containg the class labels for the training data;
##  4. value for K, the number of nearest neighbors to be used by the classifier.
library(dplyr)
Direction ~ Lag1-5 + Volume 
train.X <- training %>% 
    dplyr::select(starts_with("Lag"), Volume) %>% 
    as.matrix()

validation.X <- validation %>% 
    dplyr::select(starts_with("Lag"), Volume) %>% 
    as.matrix()

train.y <- training %>% 
    pull(Direction)

set.seed(1)
knn.fit <- knn(train = train.X, test = validation.X, 
               cl = train.y,
               k = 3)
mean(knn.fit == validation$Direction)

knn.fit <- knn(train = train.X, test = validation.X,
               cl = train.y, 
               k = 8)
mean(knn.fit == validation$Direction)

## Comparing the algorithms:
mean(validation$Direction == ifelse(logit.pred > 0.5, "Up", "Down"))
mean(validation$Direction == lda.pred$class)
mean(validation$Direction == qda.pred$class)
mean(knn.fit == validation$Direction)



##----------------------------------------------------------------------------##
## Penalized method:
library(glmnet)
glmnet

## - Here, we will use the glmnet package. The main function of this package is glmnet();
## - this function receives as input an X matrix and a Y vector;
## - We do not use the y ~ X syntax;
## 
## Let's now do a Regression example. 

## Let's predict Salary on the Hitters dataset.
basket <- Hitters
help("Hitters")
basket %>% dim

## Checking for missing values:
summary(basket)
basket %>% nrow
basket %>% na.omit %>% nrow

basket <- basket %>% na.omit()
basket

## Let's split the data into training and vvalidation set:
set.seed(27)
inTrain <- sample(nrow(basket), 
                  0.8*nrow(basket), 
                  replace = F)
training <- basket[inTrain,]
validation <- basket[-inTrain,]

training %>% str

## the model.matrix() function is particularly useful for creating X matrix. 
X.train <-  model.matrix(Salary ~ ., data = training)[, -1]
y.train <- training$Salary

X.validation <-  model.matrix(Salary ~ ., data = validation)[, -1]
y.validation <- validation$Salary


##----------------------------------------------------------------------------##
## Ridge:

## The glmnet() function has an argument called alpha;
## alpha determines what type of model is fit;
## alpha = 0 => Ridge model
library(glmnet)

grid = 10^seq(10, -2, length = 100)
ridge.fit <- glmnet(x = X.train, y = y.train, alpha = 0, lambda = grid)

## By default, the glmnet() function performs ridge for an automatically selected 
## values for lambda. 
## Also, glmnet() wil standardize the variables so they are on the same scale. 
## To turn off, use standardize = FALSE.

## Remember that for each value of lambda, there will be a set of coefficient values.
## Use the coef() function to access the 
coef(ridge.fit) %>% dim
coef(ridge.fit)[, 2]
## Higher the lambda, smaller will be the coeffients:

## Very high
ridge.fit$lambda[1]
coef(ridge.fit)[, 1]

ridge.fit$lambda[100]
coef(ridge.fit)[, 100]


## We can use predict() function for a number of purposes. 
## For instance, we can obtain the ridge coefficients for a new value of lambda
## the arguement using type = "coefficients"
predict(ridge.fit, s = 50, type = "coefficients")
## the argument "s = ..." specify the value of lambda

## To make prediction on Salary, we just provide a new dataset and a value of 
## lambda: 
predict(ridge.fit, newx = X.validation,
        s = ridge.fit$lambda[1])
predict(ridge.fit, newx = X.validation, 
        s = ridge.fit$lambda[100])

## To evaluate how well a prediction, we can use the MSE:
mean((y.validation - predict(ridge.fit, newx = X.validation, s = ridge.fit$lambda[1]))^2)
mean((y.validation - predict(ridge.fit, newx = X.validation, s = ridge.fit$lambda[100]))^2)


## We could choose the value of lambda as the one with the smalles MSE on validation set:
library(foreach)
MSE <- foreach(i = 1:length(grid), .combine = c) %do% {
    mean((y.validation - predict(ridge.fit, newx = X.validation, s = ridge.fit$lambda[i]))^2)
}
MSE %>% length
mse <- data.table(lambda = grid, mse = MSE) 
mse
mse[which.min(mse), ]

## Comparing with the MSE performance for ridge and OLS:

## OLS
mean( (validation$Salary - predict(lm(Salary ~., data = data.table(Salary = y.train, X.train)),
        data.table(X.validation)))^2)
## Ridge:
mse[which.min(mse), mse]


## checking the coefficients for the ridge regression using the best lambda:
predict(ridge.fit, type = "coefficients", s = mse[which.min(mse), lambda])



## LASSO:

## As we can see, ridge regression with a wise choice of lambda can outperform OLS;

## Now let's see if LASSO can yield either a more accurate oro a more interpretable
## model;

## alpha = 1 => LASSO

lasso.fit = glmnet(x = X.train, y = y.train, alpha = 1, lambda = grid)
plot(lasso.fit)


## we can do the same operations as before:
## We could choose the value of lambda as the one with the smalles MSE on validation set:
library(foreach)
MSE_lasso <- foreach(i = 1:length(grid), .combine = c) %do% {
    mean((y.validation - predict(lasso.fit, newx = X.validation, s = lasso.fit$lambda[i]))^2)
}
MSE_lasso %>% length
mse_lasso <- data.table(lambda = grid, mse = MSE_lasso) 
mse_lasso
mse_lasso[which.min(mse), ]

lasso.fit
plot(lasso.fit)


## Comparing:

## OLS
mean( (validation$Salary - predict(lm(Salary ~., data = data.table(Salary = y.train, X.train)),
                                   data.table(X.validation)))^2)
## Ridge:
mse[which.min(mse), mse]
## LASSO
mse_lasso[which.min(mse), mse]

predict(lasso.fit, type = "coefficients", s = mse[which.min(mse), lambda])[1:20, ]



## Elastic net:

## elastic net is a "mixing" of ridge and LASSO;
## two tunning parameter: lambda and alpha 

## 0 < alpha < 1 => elastic net

## Fitting an EN:
en.fit <- glmnet(x = X.train, y = y.train, alpha = .5, lambda = grid)

mse_en <- data.table(lambda = grid, 
                     alpha = 0.5, 
                     mse = foreach(i = 1:length(grid), .combine = c) %do% {
                         mean( (y.validation - predict(en.fit, s = en.fit$lambda[i], X.validation))^2 )
                         })

mse_en[which.min(mse)]


## Comparing:

## OLS
mean( (validation$Salary - predict(lm(Salary ~., data = data.table(Salary = y.train, X.train)),
                                   data.table(X.validation)))^2)
## Ridge:
mse[which.min(mse), mse]
## LASSO
mse_lasso[which.min(mse), mse]
## Elastic Net:
mse_en[which.min(mse), mse]


## Tunning for lambda and alpha:
mse_en <- foreach(j = seq(0.1, 0.9, by = 0.02), .combine = rbind) %do% {
    
    es.fit <- glmnet(x = X.train, y = y.train, alpha = j, lambda = grid)
    
    data.table(lambda = grid, 
               alpha = j, 
               mse = foreach(i = 1:length(grid), .combine = c) %do% {
                   mean( (y.validation - predict(es.fit, X.validation, s = en.fit$lambda[i]) )^2 )
               })
}

mse_en %>% dim

## OLS
mean( (validation$Salary - predict(lm(Salary ~., data = data.table(Salary = y.train, X.train)),
                                   data.table(X.validation)))^2)
## Ridge:
mse[which.min(mse), mse]
## LASSO
mse_lasso[which.min(mse), mse]
## Elastic Net:
mse_en[which.min(mse), mse]
mse_en[which.min(mse)]

## To check the coefficients:
predict(es.fit, 
        s = mse_en[which.min(mse), lambda], 
        alpha = mse_en[which.min(mse), alpha],
        type = "coefficients")


## For penalized logistic models
glmnet(x, y, family = "binomial", alpha = 0, lambda = NULL)
glmnet(x, y, family = "binomial", alpha = 1, lambda = NULL)
glmnet(x, y, family = "binomial", alpha = .5, lambda = NULL)

