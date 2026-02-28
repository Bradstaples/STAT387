   

rm(list = ls())


library(MASS)
library(ISLR2) # For Smarket data
library(ROCR)
attach(Smarket)

#=============================================================================================================#


train = (Year < 2005)

Smarket.2005 = Smarket[!train, ]
Direction.2005 = Direction[!train]



results.matrix = matrix(0, nrow = 5, ncol = 4)


#=============================================================================================================#

#=== KNN ===#
library(class)
train.X = cbind(Lag1 , Lag2)[train , ]
test.X  = cbind(Lag1 , Lag2)[!train , ]
train.Direction = Direction[train] # training data Y

set.seed (1)

knn.pred = knn(train.X, test.X, train.Direction , k = 3, prob = T)
table(Direction.2005, knn.pred)


knn.tn = sum((Direction.2005 == unique(Direction.2005)[1])&(knn.pred == unique(Direction.2005)[1]))
knn.tp = sum((Direction.2005 == unique(Direction.2005)[2])&(knn.pred == unique(Direction.2005)[2]))

knn.fp = sum((Direction.2005 == unique(Direction.2005)[1])&(knn.pred == unique(Direction.2005)[2]))
knn.fn = sum((Direction.2005 == unique(Direction.2005)[2])&(knn.pred == unique(Direction.2005)[1]))

knn.n = knn.tn + knn.fp
knn.p = knn.fn + knn.tp



# Specificity, Sensitivity, Overall Error/ Probability of Misclassification #
spec.knn  = 1 - (knn.fp/knn.n)
sen.knn   = knn.tp/knn.p
oer.knn1  = (knn.fn + knn.fp)/(knn.n + knn.p)
#oer.knn2  = 1 - mean(knn.pred == Direction.2005)



# ROCR #
knn.pred <- prediction(attributes(knn.pred)$prob, Smarket.2005$Direction) 
knn.perf <- performance(knn.pred, "tpr", "fpr")
plot(knn.perf, colorize = TRUE, lwd = 2)
abline(a = 0, b = 1) 


knn.auc = performance(knn.pred, measure = "auc")
#print(knn.auc@y.values)


results.matrix[1,] = as.numeric( c(spec.knn, sen.knn, oer.knn1, knn.auc@y.values))

#=============================================================================================================#

#==== LDA ===#
# availabe in the MASS library #
lda.fit = lda(Direction~Lag1+Lag2, data = Smarket, subset = train)
#lda.fit

#plot(lda.fit)

lda.pred = predict(lda.fit, Smarket.2005)
names(lda.pred)
lda.class = lda.pred$class

#table(lda.class, Direction.2005)
table(Direction.2005, lda.class)


# sum(lda.pred$posterior[,1] >= 0.5) 
# sum(lda.pred$posterior[,1] < 0.5)  



lda.tn = sum((Direction.2005 == unique(Direction.2005)[1])&(lda.class == unique(Direction.2005)[1]))
lda.tp = sum((Direction.2005 == unique(Direction.2005)[2])&(lda.class == unique(Direction.2005)[2]))

lda.fp = sum((Direction.2005 == unique(Direction.2005)[1])&(lda.class == unique(Direction.2005)[2]))
lda.fn = sum((Direction.2005 == unique(Direction.2005)[2])&(lda.class == unique(Direction.2005)[1]))

lda.n = lda.tn + lda.fp
lda.p = lda.fn + lda.tp
  
  

# Specificity, Sensitivity, Overall Error/ Probability of Misclassification #

spec.lda  = 1 - (lda.fp/lda.n)
sen.lda   = lda.tp/lda.p
oer.lda1  = (lda.fn + lda.fp)/(lda.n + lda.p)
#oer.lda2  = 1 - mean(lda.class == Direction.2005)



# ROCR #

lda.pred <- prediction(lda.pred$posterior[,2], Smarket.2005$Direction) 
lda.perf <- performance(lda.pred,"tpr","fpr")
plot(lda.perf,colorize=TRUE, lwd = 2)
abline(a = 0, b = 1) 


lda.auc = performance(lda.pred, measure = "auc")
#print(lda.auc@y.values)



results.matrix[2,] = as.numeric( c(spec.lda, sen.lda, oer.lda1, lda.auc@y.values))

#=============================================================================================================#

#=== QDA ===#

qda.fit = qda(Direction~Lag1+Lag2, data = Smarket, subset = train)
#qda.fit


qda.pred  = predict(qda.fit, Smarket.2005)
qda.class = qda.pred$class

table(qda.class, Direction.2005)


qda.tn = sum((Direction.2005 == unique(Direction.2005)[1])&(qda.class == unique(Direction.2005)[1]))
qda.tp = sum((Direction.2005 == unique(Direction.2005)[2])&(qda.class == unique(Direction.2005)[2]))

qda.fp = sum((Direction.2005 == unique(Direction.2005)[1])&(qda.class == unique(Direction.2005)[2]))
qda.fn = sum((Direction.2005 == unique(Direction.2005)[2])&(qda.class == unique(Direction.2005)[1]))

qda.n = qda.tn + qda.fp
qda.p = qda.fn + qda.tp



# Specificity, Sensitivity, Overall Error/ Probability of Misclassification #
spec.qda  = 1 - (qda.fp/qda.n)
sen.qda   = qda.tp/qda.p
oer.qda1  = (qda.fn + qda.fp)/(qda.n + qda.p)
#oer.qda2  = 1 - mean(qda.class == Direction.2005)



# ROCR #
qda.pred <- prediction(qda.pred$posterior[,2], Smarket.2005$Direction) 
qda.perf <- performance(qda.pred,"tpr","fpr")
plot(qda.perf,colorize=TRUE, lwd = 2)
abline(a = 0, b = 1) 


qda.auc = performance(qda.pred, measure = "auc")
print(qda.auc@y.values)


results.matrix[3,] = as.numeric( c(spec.qda, sen.qda, oer.qda1, qda.auc@y.values))


#=============================================================================================================#

#=== Logistic Regression ===#
# pairs(Smarket, upper.panel = NULL)
# 
# log.reg_try = glm(Direction ~., data = Smarket[train,],
#               family = binomial, subset = train)

log.reg = glm(Direction ~ Lag1 + Lag2, data = Smarket[train,], 
              family = binomial, subset = train)

#summary(log.reg)

log.probs = predict(log.reg, Smarket.2005, type = "response")
length(log.probs)
head(log.probs)
#contrasts(Direction)

log.pred = rep("Down",252)
log.pred[log.probs > 0.5] = "Up"
tail(log.pred)


table(Direction.2005, log.pred)


log.tn = sum((Direction.2005 == unique(Direction.2005)[1])&(log.pred == unique(Direction.2005)[1]))
log.tp = sum((Direction.2005 == unique(Direction.2005)[2])&(log.pred == unique(Direction.2005)[2]))

log.fp = sum((Direction.2005 == unique(Direction.2005)[1])&(log.pred == unique(Direction.2005)[2]))
log.fn = sum((Direction.2005 == unique(Direction.2005)[2])&(log.pred == unique(Direction.2005)[1]))

log.n = log.tn + log.fp
log.p = log.fn + log.tp



# Specificity, Sensitivity, Overall Error/ Probability of Misclassification #

spec.log  = 1 - (log.fp/log.n)
sen.log   = log.tp/log.p
oer.log1  = (log.fn + log.fp)/(log.n + log.p)
#oer.log2  = 1 - mean(log.pred == Direction.2005)



# ROCR #
log.pred <- prediction(log.probs, Smarket.2005$Direction) 
log.perf <- performance(log.pred,"tpr","fpr")
plot(log.perf,colorize=TRUE, lwd = 2)
abline(a = 0, b = 1) 


log.auc = performance(log.pred, measure = "auc")
#print(log.auc@y.values)

results.matrix[4,] = as.numeric( c(spec.log, sen.log, oer.log1, log.auc@y.values))


#=============================================================================================================#

# Naive Bayes #

library(e1071)

nb.fit = naiveBayes(Direction ~ Lag1 + Lag2, data = Smarket, subset = train)
nb.fit

nb.class = predict(nb.fit , Smarket.2005)
table(Direction.2005, nb.class)

#mean(nb.class == Direction.2005)


nb.tn = sum((Direction.2005 == unique(Direction.2005)[1])&(nb.class == unique(Direction.2005)[1]))
nb.tp = sum((Direction.2005 == unique(Direction.2005)[2])&(nb.class == unique(Direction.2005)[2]))

nb.fp = sum((Direction.2005 == unique(Direction.2005)[1])&(nb.class == unique(Direction.2005)[2]))
nb.fn = sum((Direction.2005 == unique(Direction.2005)[2])&(nb.class == unique(Direction.2005)[1]))

nb.n = nb.tn + nb.fp
nb.p = nb.fn + nb.tp




# Specificity, Sensitivity, Overall Error/ Probability of Misclassification #
spec.nb  = 1 - (nb.fp/nb.n)
sen.nb   = nb.tp/nb.p
oer.nb1  = (nb.fn + nb.fp)/(nb.n + nb.p)
#oer.nb2  = 1 - mean(nb.class == Direction.2005)


# ROCR #
nb.pred = predict(nb.fit , Smarket.2005, type = "raw")
nb.pred <- prediction(nb.pred[,2], Smarket.2005$Direction) 
nb.perf <- performance(nb.pred,"tpr","fpr")
plot(nb.perf,colorize=TRUE, lwd = 2)
abline(a = 0, b = 1) 


nb.auc = performance(nb.pred, measure = "auc")
#print(nb.auc@y.values)


results.matrix[5,] = as.numeric( c(spec.nb, sen.nb, oer.nb1, nb.auc@y.values))


colnames(results.matrix) = c("SPEC", "SENS", "OER", "AUC")
rownames(results.matrix) = c("KNN", "LDA", "QDA", "LOG", "NAIVE-B")

results.matrix
#=============================================================================================================#






