library(ISLR)
attach(Auto)
#simple linear regression withj mpg as response and horsepower as predictor
reg1<-lm(mpg~horsepower,data=Auto)
summary(reg1)
#predicted mpg associated with a horsepower of 98
predict(reg1,data.frame(horsepower=c(98)),interval="prediction")
#associated 95% coinfidence and prediction intervals for a horsepower of 98
predict(reg1,data.frame(horsepower=c(98)),interval="confidence")
predict(reg1,data.frame(horsepower=c(98)),interval="prediction")
#plotting linear model
plot(horsepower,mpg)
abline(reg1,col="red")
#Use the plot() function to produce diagnostic plots of the least
#squares regression fit. Comment on any problems you see with
#the fit.
plot(reg1)

attach(Carseats)
summary(Carseats)
help(Carseats)
reg2<-lm(Sales~Price+Urban+US)
summary(reg2)
reg3<-lm(Sales~Price+US)
summary(reg3)

set.seed(1)
#using rnorm function create a vector x containting 100 observations frwan from a n(0,1) distribuition
x<-rnorm(100)
head(x)
#creating a vector eps containing 100 observations from a n(0,0.25) distribution
eps<-rnorm(100,0,0.25)
head(eps)
#using x and eps create a vector y accoding to the model y=-1+0.5x+error
y<- -1 + 0.5*x + eps
length(y)
head(y)
#scatterplot of x against y
plot(x,y)
# plot model for x and y
reg4<-lm(y~x)
summary(reg4)
