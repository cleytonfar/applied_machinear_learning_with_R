## Loading packages:
library(data.table)
library(ggplot2)
library(ISLR)
library(tidyr)
library(dplyr)
library(tree)

## Let's use a data set containing sales of child car seats;
foo <- data.table(Carseats)
foo %>% dim

## This dataset has 11 variables:

## Sales: Unit sales at each location
## CompPrice: Price charged by competitor at each location
## Income: Community income level
## Advertising: Local advertising budget for company at each location 
## Price: Price company charges for car seats at each site
## ShelveLoc: factor with levels Bad, Good and Medium indicating the quality of 
##            the shelving location for the car seats at each site
## Education: aEducation level at each location


## Our Job: Predict if the Sales will be high or no.

## Let'see the type of the variables:
foo %>% str

## Let's create a categorical variable that indicates the if Sales were high or not:
foo$High <- ifelse(foo$Sales > 8, "Yes", "No")
foo <- data.table(mutate(foo, High = as.factor(High)))


##----------------------------------------------------------------------------##
## Separating the training and test set:

## creating an index for obs that will be included in training set:
set.seed(10)
inTrain <- sample(nrow(foo), size = 0.8*nrow(foo), replace = F)

## Now we separate the two sets:
training <- foo[inTrain]
testing<- foo[-inTrain]

##----------------------------------------------------------------------------##
## DECISION TREES:

# To fit a classification tree, we use the tree() function from the tree library:
library(tree)
tree.fit <- tree(formula = High ~.-Sales, 
                 data = foo)
tree.fit

summary(tree.fit)

# We can plot the tree by using the plot() function on the tree.fit:
plot(tree.fit, col = 'blue', type = "proportional")
title("Tree representation")
text(tree.fit, pretty = 0, cex = 0.8, col = 'red')


# Let's check the performance on test set:
tree.pred <- predict(object = tree.fit, 
                     newdata = testing,
                     type = "class") 


## Accuracy:
mean(testing$High == tree.pred)
table(testing$High, tree.pred)


## Now, let's consider whether prunning the tree might lead to some improvement.
## To perform cost complexity prunning, prune.misclass() function.
## Here is different:

## 1. A large and complex tree must be already be estimated;
## 2. on this tree, we apply the cost complexity prunning;

## prune.misclass() has two main arguments:
    ## 1. tree = "..." the tree already grown
    ## 2. k = "...". This is the value for alpha


tree.pruned.fit <- prune.misclass(tree = tree.fit, k = 9)
mean(testing$High == predict(tree.pruned.fit, testing, type = "class"))

tree.pruned.fit <- prune.misclass(tree = tree.fit, k = 4)
mean(testing$High == predict(tree.pruned.fit, testing, type = "class"))

tree.pruned.fit <- prune.misclass(tree = tree.fit, k = 2)
mean(testing$High == predict(tree.pruned.fit, testing, type = "class"))

library(foreach)
perf <- data.table(alpha = seq(0, 3, by = 0.01),
                   acc = foreach(i = seq(0, 3, by = 0.01),
                                 .combine = 'c') %do% {
                                     mean(testing$High == predict(prune.misclass(tree = tree.fit, k = i), 
                                                                  testing,
                                                                  type = "class"))})
perf
perf[which.max(acc)]

final.tree.fit <- prune.misclass(tree.fit, k = 1)

plot(final.tree.fit, col = 'blue', type = "proportional")
title("Pruned Tree Representation")
text(final.tree.fit, pretty = 0, cex = 0.8, col = 'red')


##----------------------------------------------------------------------------##
## BAGGING and RANDOM FOREST:

## Let's apply bagging and Random Forest to a more challeging problem: predicting
## House Market.

## Here, let's use the Boston dataset.
library(MASS) # to load the Boston dataset
house <- data.table(Boston)

## Boston dataset is a set of house values in Suburbs of Boston. It contains 
## 506 rows and 14 columns. Some of the columns are:
##  crim: per capita crime rate by town.
##  zn: roportion of residential land zoned for lots over 25,000 sq.ft.
##  indus: non-retail business acres per town.
##  medv: median value of owner-occupied homes in \$1000s.

house %>% str


##----------------------------------------------------------------------------##
## First things first: separate the original dataset into training and test set
set.seed(10)
inTrain <- sample(x = nrow(house), size = 0.8*nrow(house), replace = F)
training <- house[inTrain]
testing <- house[-inTrain]
##----------------------------------------------------------------------------##

## We apply bagging and Random Forest using the randomForest library. 
## Remember that bagging involves take draws from the original dataset using 
## bootstrap and then running unpruned decision tree to each draw. 
## On the other hand, Random Forest is just an extension of bagging in which tries to 
## decorrelates the decision trees by sampling the column considered at each split.
## Therefore, we can run both model using the same function: randomForest() function.
library(randomForest)

## We perform bagging using randomForest() function but setting the argument 
## mtry = ... equal to the number of columns in the dataset. 

bag.fit <- randomForest(medv ~ .,
                        data = training, 
                        mtry = ncol(training) - 1,
                        ntree = 50,
                        importance = T)

bag.fit

bag.pred <- predict(bag.fit, testing)
mean((testing$medv - bag.pred)^2)

bag.fit <- randomForest(medv ~ .,
                        data = training, 
                        mtry = ncol(training) - 1,
                        ntree = 200,
                        importance = T)

bag.pred <- predict(bag.fit, testing)
mean((testing$medv - bag.pred)^2)


bag.fit <- randomForest(medv ~ .,
                        data = training, 
                        mtry = ncol(training) - 1,
                        ntree = 500,
                        importance = T)

bag.pred <- predict(bag.fit, testing)
mean((testing$medv - bag.pred)^2)


## Growing a randomForest is exactly the same way, except that we use a smaller
## value of the mtry = "..." argument.
set.seed(10)
rf.fit <- randomForest(medv ~.,
                       data = training, 
                       mtry = sqrt(ncol(training)),
                       ntree = 10,
                       importance = T)

rf.pred <- predict(rf.fit, testing)
mean((testing$medv - rf.pred)^2)


rf.fit <- randomForest(medv ~.,
                       data = training, 
                       mtry = sqrt(ncol(training)),
                       ntree = 100,
                       importance = T)

rf.pred <- predict(rf.fit, testing)
mean((testing$medv - rf.pred)^2)

plot(density(testing$medv))
lines(density(rf.pred))

## Using the importance() function, we can view the importance of each variable.
importance(rf.fit) %>% 
    as.data.table(keep.rownames = "var") %>% 
    arrange(-IncNodePurity)

## Two measures are reported. 
## %IncMSE: mean increase of MSE in prediction when a given variable is excluded
## IncNodePurity: total increase in node purity when a given variable is excluded

## We can also plot these results:
varImpPlot(rf.fit)


##----------------------------------------------------------------------------##
## Boosting:

## Here we use the gbm package:
library(gbm)

## In order to fit a boosted regression trees to the boston data, we use the gbm()
## function. We run the gbm() with the option distribution = "gaussian" since this
## is a regression problem. For a classification binary problem, we would use
## distribution = "bernoulli".
## n.trees = 5000 indicates that we want 5000 trees, and the 
## option interaction.depth = 4 limits the depth of each tree to 4.
## The shrinkage is the parameter that controls the learning rate.

set.seed(1)

gbm.fit <- gbm(medv ~ .,
               data = training, 
               distribution = "gaussian",
               n.trees = 1000, 
               shrinkage = 0.01,
               interaction.depth = 4)

summary(gbm.fit)

## This last command will report a relative influence plot and also outputs the 
## relative influence statistics.

## To make a prediction, we use the predict() function. Also, we must specify
## the n.trees = "..." parameters:
gbm.pred <- predict(gbm.fit, testing, n.trees = 1000)

## performance:
mean( (testing$medv - gbm.pred)^2 )


##----------------------------------------------------------------------------##
## SVM

library(e1071)

# Support Vector CLASSIFIER:

# We will use the function svm() in order to perform the support vector classifier.
# To this end, we will use the option kernel = "linear".

# Let's generate the data:
set.seed(1)
x = matrix(rnorm(2000*2), ncol = 2)
y = c(rep(-1, 1000), rep(1, 1000))
x[y==1, ] = x[y==1, ] + 1

# Let's plot the points to see if the classes are well separable:
ggplot(data = data.table(x, y)) + 
    aes(x = V1, y = V2, col = factor(y)) + 
    geom_point() +
    theme_minimal() + 
    labs(color = "")

# Now, in order to fit a SVC, we must encode the response as a factor variable:
foo <- data.table(x, y = as.factor(y))


##----------------------------------------------------------------------------##
## 
inTrain <- sample(nrow(foo), 0.8*nrow(foo), replace = F)
training <- foo[inTrain]
testing <- foo[-inTrain]
##----------------------------------------------------------------------------##

# Fitting the SVC:
svc.fit <- svm(formula = y~., 
               data = training, 
               kernel = "linear", 
               cost = 10, 
               scale = F)

# The argument scale = F tells the svm() function not to scale the variable
# to have mean zero and standard deviation one.

mean(testing$y == predict(svc.fit, testing))


# In order to perform a Support Vector Machine model, we once again
# use the svm() function, but with a different kernel. 
# To fit a svm() with polynomial kernel,  we choose the kernel = "polynomial"
## 
# To fit a svm() with radial kernel, we choose the kernel = "radial".


## Generate data:
set.seed(1)
x = matrix(rnorm(200*2), ncol = 2)
x[1:100, ] = x[1:100, ] + 2
x[101:150, ] = x[101:150, ] + -2
y = c(rep(1,150), rep(2, 50))
dat  = data.table(x, y = as.factor(as.character(y)))
plot(x, col = (y), pch = 19)

##----------------------------------------------------------------------------##
inTrain = sample(200, 0.8*200, replace = F)
training <- dat[inTrain]
testing <- dat[-inTrain]
##----------------------------------------------------------------------------##



##----------------------------------------------------------------------------##
## SVM kernel radial:
svm.radial.fit <- svm(y ~ .,
                      data = training,
                      kernel = "radial", 
                      gamma = 1, 
                      cost = 1)

plot(svm.radial.fit, training)

## Performance:
mean(testing$y == predict(svm.radial.fit, testing))


##----------------------------------------------------------------------------##
## SVM poly
svm.poly.fit <- svm(y ~ .,
                    data = training,
                    kernel = "polynomial", 
                    d = 3, 
                    cost = 3)

plot(svm.poly.fit, training)

mean(testing$y == predict(svm.poly.fit, testing))




##----------------------------------------------------------------------------##
## INTRODUCTION TO DEEP LEARNING

library(keras)

imdb <- dataset_imdb(num_words = 5000)

c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb

## Preparing the data:
## You can feed lists of integers into a neural network. You have to turn your
## lists into tensors. Let's perform one-hot encode to turn them into vector:

vectorize_sequence <- function(sequence, dimension = 5000){
    
    ## Create a matrix of zeros:
    results <- matrix(0, nrow = length(sequence), ncol = dimension)
    
    ## Loop to include 1 into the respective column:
    for(i in 1:length(sequence)){
        results[i, sequence[[i]]] <- 1 
        ## sequence[[i]] return a vecot of numbers
    }
    
    results
}

x_train <- vectorize_sequence(train_data)
x_test <- vectorize_sequence(test_data)

y_train <- train_labels %>% as.numeric
y_test <- test_labels %>% as.numeric

## Now the data is ready to be fed into a neural network.


## Now let's define our model. There are two key architecture decisions to be 
## made about suuch a stack of dense layers
##  1. How many layers to use;
##  2. How many hidden units to choose for each layers;

## Let's stick with the following architecture:
model <- keras_model_sequential() %>% 
    layer_dense(units = 16, activation = "relu", input_shape = ncol(x_train)) %>% 
    layer_dense(units = 16, activation = "relu") %>% 
    layer_dense(units = 1, activation = "sigmoid")


## Finally we need to choose the LOSS FUNCTION and an OPTMIZER. 
## Because we are facing a binary classification problem and the output of the 
## network will be a probability, it is best to use the binary_crossentropy.

model %>% 
    compile(loss = "binary_crossentropy", 
            optimizer = "rmsprop", 
            metrics = "accuracy")

## NOTE: compile() function modifies the network in place rather than return an 
##       object.


## Validating approach:
## In order to monitor during training the accuracy of the model, we will create 
## a validation set by setting apart 10.000 samples from the original train data:
val_indices <- 1:10000

x_val <- x_train[val_indices, ]
y_val <- y_train[val_indices]

partial_x_train <- x_train[-val_indices, ]
partial_y_train <- y_train[-val_indices]


## We will now train the model for 20 epochs, in mini-batches of 512 samples. 

history <- model %>% 
    fit(x = partial_x_train,
        y = partial_y_train,
        batch_size = 512, 
        epochs = 20, 
        validation_data = list(x_val, y_val)
    )


str(history)

history 

## As we can see, the training loss decreases with every epoch, and the training
## accuracy increases with every epoch. But that is not the case for the 
## validation loss and accuracy: They seem to peak at the 5th epoch. 


## Let's train a new network from scratch for 5 epochs and then evaluate it on the
## test data:

model <- keras_model_sequential() %>% 
    layer_dense(units = 16, activation = "relu", input_shape = ncol(x_train)) %>% 
    layer_dense(units = 16, activation = "relu") %>% 
    layer_dense(units = 1, activation = "sigmoid")


model %>% 
    compile(optimizer = "rmsprop", 
            loss = "binary_crossentropy", 
            metrics = "accuracy")

model %>% fit(x = x_train,
              y = y_train, 
              epochs = 5, 
              batch_size = 512)

results <- model %>% evaluate(x_test, y_test)
results



## Using a trained network to generate prediction on new data:

## After training a network, you will want to use it in a practical setting.
model %>% predict(x_test[1:10, ])




##----------------------------------------------------------------------------##
## Another example:

## MNIST dataset
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y

x_test <- mnist$test$x
y_test <- mnist$test$y

class(x_train)
class(y_train)

class(x_test)
class(y_test)

x_train %>% dim
y_train %>% dim

## The x data is a 3-d array (images, width, height) of grayscale values . To 
## prepare the data for training we convert the 3-d arrays into matrices by 
## reshaping width and height into a single dimension (28x28 images are 
## flattened into length 784 vectors). 
x_train <- array_reshape(x_train, dim = c(nrow(x_train), 28*28))
x_test <- array_reshape(x_test, dim = c(nrow(x_test), 28*28))

dim(x_train)
dim(x_test)

## Then, we convert the grayscale values from integers ranging between 0 to 255 
## into floating point values ranging between 0 and 1:
x_train <- x_train / 255 ## scaling
x_test <- x_test / 255 ## scaling

## The y data is an integer vector with values ranging from 0 to 9. To prepare 
## this data for training we use one-hot encode the vectors into binary class 
## matrices using the Keras to_categorical() function:
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

y_train %>% dim
y_test %>% dim

## Defining the Model:
## The core data structure of Keras is a model, a way to organize layers. The 
## simplest type of model is the Sequential model, a linear stack of layers.

## We begin by creating a sequential model and then adding layers using the pipe 
## (%>%) operator:
model <- keras_model_sequential()

model %>% 
    layer_dense(units = 256, activation = "relu", input_shape = c(28*28)) %>% 
    layer_dropout(rate = 0.4) %>% 
    layer_dense(units = 128, activation = "relu") %>% 
    layer_dropout(rate = 0.3) %>% 
    layer_dense(units = 10, activation = "softmax")

## The input_shape argument to the first layer specifies the shape of the input 
## data (a length 784 numeric vector representing a grayscale image). The final 
## layer outputs a length 10 numeric vector (probabilities for each digit) using
## a softmax activation function.

## Use the summary() function to print the details of the model:
summary(model)

## Next, to make our network ready to train we need to pick three more things: 
##  1. loss function;
##  2. optimizer;
##  3. metric to monitor during training and test;

## After that, we compile the model with appropriate loss function, optimizer, 
## and metrics:
model %>% 
    compile(loss = "categorical_crossentropy",
            optimizer = optimizer_rmsprop(),
            metrics = "accuracy")

## NOTE: compile() function modifies the network in place rather than return an 
##       object.


## Training and Evaluation
## Use the fit() function to train the model for 30 epochs using batches of 128 
## images:
history <- model %>% 
    fit(x_train, y_train, 
        epochs = 30,
        batch_size = 128, 
        validation_split = 0.2)

## The history object returned by fit() includes loss and accuracy metrics which 
## we can plot:
plot(history)

## Evaluate the model’s performance on the test data:
model %>% 
    evaluate(x_test, y_test)

## Generate predictions on new data:
model %>% 
    predict_classes(x_test)

