rm(list = ls())

##################
#=== LIBRARIES===#
##################


library(ISLR2)
library(boot) # LOOCV


#####################################################################
attach(Auto)
str(Auto)

n.len = nrow(Auto)

set.seed(1)

train = sample(n.len, n.len/2)

Valid.testX = Auto[-train,]
Valid.testY = mpg[-train]

#==========================================================================================#

#=== Cross Validation ===#

# Validation Set #

Valid.lm1 = lm(mpg ~ horsepower, data = Auto, subset = train)
Valid.pred1 = predict(Valid.lm1, Valid.testX)

Valid.MSE1 = mean((Valid.testY - Valid.pred1)^2) # Estimate Test MSE



Valid.lm2 = lm(mpg ~ poly(horsepower, 2), data = Auto, subset = train)
Valid.pred2 = predict(Valid.lm2, Valid.testX)

Valid.MSE2 = mean((Valid.testY - Valid.pred2)^2) # Estimate Test MSE




Valid.lm3 = lm(mpg ~ poly(horsepower, 3), data = Auto, subset = train)
Valid.pred3 = predict(Valid.lm3, Valid.testX)

Valid.MSE3 = mean((Valid.testY - Valid.pred3)^2) # Estimate Test MSE

#==========================================================================================#


# LOOCV #

LOOCV.lm1 = glm(mpg ~ horsepower, data = Auto)
summary(LOOCV.lm1)
coef(LOOCV.lm1)


loocv.err = cv.glm(Auto, LOOCV.lm1)$delta

loocv.err = rep(0,10)
for(i in 1:10){
  
  LOOCV.lm1 = glm(mpg ~ poly(horsepower, i), data = Auto)
  
  LOOCCV.err[i] = cv.glm(Auto, LOOCV.lm1)$delta[1]
  
}

LOOCCV.err # Estimate Test MSE


#==========================================================================================#

# K-Fold CV #
set.seed(17)

cv.err10 = rep(0,10)

for(i in 1:10){
  
  cv.err10.lm1 = glm(mpg ~ poly(horsepower, i), data = Auto)
  cv.err10[i] = cv.glm(Auto, cv.err10.lm1, K = 10)$delta[1]
  
}

cv.err10




#####################################################################
