#This question should be answered using the Weekly data set, which
#is part of the ISLR2 package. This data is similar in nature to the
#Smarket data from this chapter’s lab, except that it contains 1, 089
#weekly returns for 21 years, from the beginning of 1990 to the end of
#2010.

library(ISLR2)
library(glmnet)
library(caret)
attach(Weekly)
#a)Produce some numerical and graphical summaries of the Weekly
#data. Do there appear to be any patterns?
summary(Weekly)
pairs(Weekly)
#b)Use the full data set to perform a logistic regression with
#Direction as the response and the five lag variables plus Volume
#as predictors. Use the summary function to print the results. 
mod1 <- glm(Direction ~ Lag1+Lag2+Lag3+Lag4+Lag5+Volume, data = Weekly, family="binomial")
summary(mod1)
#(c) Compute the confusion matrix and overall fraction of correct
#predictions. 
modProbs <- predict(mod1, type = "response")
modPred <- ifelse(modProbs > 0.5, "Up", "Down")
table(modPred, Weekly$Direction)
mean(modPred == Weekly$Direction)
actualDirection <- Weekly$Direction
predsFactor <- factor(modPred, levels = c("Down", "Up"))
actualDirectionFactor <- factor(actualDirection, levels = c("Down", "Up"))
confusionMatrix(predsFactor, actualDirectionFactor)
#(d) Now fit the logistic regression model using a training data period
#from 1990 to 2008, with Lag2 as the only predictor. Compute the
#confusion matrix and the overall fraction of correct predictions
#for the held out data (that is, the data from 2009 and 2010).
train <- (Weekly$Year < 2009)
mod2 <- glm(Direction ~ Lag2, data = Weekly, family = binomial, subset = train)
mod2Prob <- predict(mod2, Weekly[!train, ], type = "response")
mod2Pred <- ifelse(mod2Prob > 0.5, "Up", "Down")
table(mod2Pred, Weekly$Direction[!train])
mean(mod2Pred == Weekly$Direction[!train])
predsFactor2 <- factor(mod2Pred, levels = c("Down", "Up"))
actualDirectionFactor2 <- factor(Weekly$Direction[!train], levels = c("Down", "Up"))
confusionMatrix(predsFactor2, actualDirectionFactor2)

#(e) Repeat (d) using LDA.
library(MASS)
lda.fit <- lda(Direction ~ Lag2, data = Weekly, subset = train)
lda.pred <- predict(lda.fit, Weekly[!train, ])
table(lda.pred$class, Weekly$Direction[!train])
mean(lda.pred$class == Weekly$Direction[!train])
predsFactorLDA <- factor(lda.pred$class, levels = c("Down", "Up"))
actualDirectionFactorLDA <- factor(Weekly$Direction[!train], levels = c("Down", "Up"))
confusionMatrix(predsFactorLDA, actualDirectionFactorLDA)

#(f) Repeat (d) using QDA.
qda.fit <- qda(Direction ~ Lag2, data = Weekly, subset = train)
qda.pred <- predict(qda.fit, Weekly[!train, ])
table(qda.pred$class, Weekly$Direction[!train])
mean(qda.pred$class == Weekly$Direction[!train])
predsFactorQDA <- factor(qda.pred$class, levels = c("Down", "Up"))
actualDirectionFactorQDA <- factor(Weekly$Direction[!train], levels = c("Down", "Up"))
confusionMatrix(predsFactorQDA, actualDirectionFactorQDA)

#(g) Repeat (d) using KNN with K = 1.
library(class)
train.X <- as.matrix(Weekly$Lag2[train])
test.X <- as.matrix(Weekly$Lag2[!train])
train.Direction <- Weekly$Direction[train]
set.seed(1)
knn.pred <- knn(train.X, test.X, train.Direction, k = 1)
table(knn.pred, Weekly$Direction[!train])
mean(knn.pred == Weekly$Direction[!train])
predsFactorKNN <- factor(knn.pred, levels = c("Down", "Up"))
actualDirectionFactorKNN <- factor(Weekly$Direction[!train], levels = c("Down", "Up"))
confusionMatrix(predsFactorKNN, actualDirectionFactorKNN)

#(h) Repeat (d) using naive Bayes
library(e1071)
nb.fit <- naiveBayes(Direction ~ Lag2, data = Weekly, subset = train)
nb.pred <- predict(nb.fit, Weekly[!train, ])
table(nb.pred, Weekly$Direction[!train])
mean(nb.pred == Weekly$Direction[!train])
predsFactorNB <- factor(nb.pred, levels = c("Down", "Up"))
actualDirectionFactorNB <- factor(Weekly$Direction[!train], levels = c("Down", "Up"))
confusionMatrix(predsFactorNB, actualDirectionFactorNB)


#Question 8
#8)a) In this dataset what is n and what is p? Write out the model used to generate the data in equation form.
set.seed(1)
x <- rnorm(100)
y <- x - 2 * (x^2) + rnorm(100)
n <- length(y) # n = 100
p <- 2 # p = 2 (x and x^2)
n
p
# Model: y = β0 + β1*x + β2*x^2 + ε
plot(x, y)

#c)set a random seed, and then compute the LOOCV errors that result from fitting the following four models:
#i) Y = β0 + ε
#ii) Y = β0 + β1X + ε
#iii) Y = β0 + β1X + β2X^2 + ε
#iv) Y = β0 + β1X + β2X^2 + β3X^3 + ε
set.seed(1)
loocv_errors <- sapply(1:4, function(degree) {
  cv_error <- 0
  for (i in 1:n) {
    train_x <- x[-i]
    train_y <- y[-i]
    test_x <- x[i]
    test_y <- y[i]
    model <- lm(train_y ~ poly(train_x, degree))
    pred <- predict(model, newdata = data.frame(train_x = test_x))
    cv_error <- cv_error + (pred - test_y)^2
  }
  return(cv_error / n)
})
names(loocv_errors) <- c("Degree 0", "Degree 1", "Degree 2", "Degree 3")
loocv_errors
