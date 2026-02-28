install.packages("ISLR")
library(ISLR)
data("Auto")
summary(Auto)
help(Auto)
attach(Auto)
library(car)
# Remove rows with missing values
Auto <- na.omit(Auto)
#range of each quantitative predictor
range(mpg)
range(horsepower)
range(weight)
range(acceleration)
range(year)
range(cylinders)
range(displacement)
#mean and standard deviation of each predictor
mean(mpg)
sd(mpg)
mean(cylinders)
sd(cylinders)
mean(displacement)
sd(displacement)
mean(horsepower)
sd(horsepower)
mean(weight)
sd(weight)
mean(acceleration)
sd(acceleration)
mean(year)
sd(year)

#remove 10th through 85th observations
Auto_subset <- Auto[-(10:85), ]
summary(Auto_subset)
#range, stand deviation and mean of quantatative predictors for the subset
range(Auto_subset$mpg)
sd(Auto_subset$mpg)
mean(Auto_subset$mpg)

range(Auto_subset$cylinders)
sd(Auto_subset$cylinders)
mean(Auto_subset$cylinders)

range(Auto_subset$displacement)
sd(Auto_subset$displacement)
mean(Auto_subset$displacement)

range(Auto_subset$horsepower)
sd(Auto_subset$horsepower)
mean(Auto_subset$horsepower)

range(Auto_subset$weight)
sd(Auto_subset$weight)
mean(Auto_subset$weight)

range(Auto_subset$acceleration)
sd(Auto_subset$acceleration)
mean(Auto_subset$acceleration)

range(Auto_subset$year)
sd(Auto_subset$year)
mean(Auto_subset$year)

plot(Auto)
scatterplotMatrix(Auto)
pairs(Auto)
model <- lm(mpg ~ . - name, data = Auto)
crPlots(model)
plot(model)
#horsepower plots
plot(horsepower, mpg, main="Horsepower vs. MPG", xlab="Horsepower", ylab="Miles per Gallon (MPG)", pch=19)
plot(horsepower, displacement, main="Horsepower vs. Displacement", xlab="Horsepower", pch=19)
plot(horsepower, weight, main="Horsepower vs. Weight", xlab="Horsepower", ylab="Weight", pch=19)
plot(horsepower, acceleration, main="Horsepower vs. Acceleration", xlab="Horsepower", ylab="Acceleration", pch=19)

#plots for mpg vs weight, displacement, horsepower
plot(weight, mpg, main="Weight vs. MPG", xlab="Weight", ylab="Miles per Gallon (MPG)", pch=19)
plot(displacement, mpg, main="Displacement vs. MPG", xlab="Displacement", ylab="Miles per Gallon (MPG)", pch=19)
plot(horsepower, mpg, main="Horsepower vs. MPG", xlab="Horsepower", ylab="Miles per Gallon (MPG)", pch=19)
plot(cylinders, mpg, main="Cylinders vs. MPG", xlab="Cylinders", ylab="Miles per Gallon (MPG)", pch=19)
plot(year, mpg, main="Year vs. MPG", xlab="Year", ylab="Miles per Gallon (MPG)", pch=19)


pnorm(-6.17)
