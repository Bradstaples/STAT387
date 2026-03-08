library(leaps)
library(class)
library(caret)
library(lmtest)
library(glmnet)
library(ggplot2)
library(corrplot)
library(randomForest)
library(gbm)
library(e1071)
library(ISLR2)
library(keras3)
library(tensorflow)
###############################################################################
# 1. Load with factors
bank <- read.csv("bank_marketing.csv", sep = ";", stringsAsFactors = FALSE)
summary(bank)
#change the y column so yes = 1 and no = 0
bank$y <- ifelse(bank$y == "yes", 1, 0)
#############################################################################
#exploratory data analysis
table(bank$y)
prop.table(table(bank$y))
#plot(bank)
#pairs(bank)
#############################################################################
# Calculate means of numeric variables grouped by the response y
groupMeans <- aggregate(cbind(age, balance, duration, campaign) ~ y, 
                        data = bank, FUN = mean)
print("Mean values for subscribers (1) vs non-subscribers (0):")
print(groupMeans)
#############################################################################
# Response rate by Job type
jobResponse <- tapply(bank$y, bank$job, mean)
jobResponse <- sort(jobResponse, decreasing = TRUE)
print("Conversion rate by Job (sorted):")
print(round(jobResponse, 4))
#############################################################################
# Response rate by Education level
eduRepsonse <- tapply(bank$y, bank$education, mean)
print("Conversion rate by Education:")
print(round(eduRepsonse, 4))
#############################################################################
#overlaid density plots
ggplot(bank, aes(x=age, fill=factor(y))) +
  geom_density(alpha=0.5) +
  labs(title="Density Plot of Age by Response Variable",
       x="Age",
       fill="Response (y)") +
  theme_minimal()
#############################################################################
#does duration separate responses
ggplot(bank, aes(x=duration, fill=factor(y))) +
  geom_density(alpha=0.5) +
  labs(title="Density Plot of Duration by Response Variable",
       x="Duration(Seconds)",
       fill="Response (y)") +
  theme_minimal()
#############################################################################
#does age separate responses
ggplot(bank, aes(x=age, fill=factor(y))) +
  geom_density(alpha=0.5) +
  labs(title="Density Plot of Age by Response Variable",
       x="Age",
       fill="Response (y)") +
  theme_minimal()
#############################################################################
#mosaic plot of response and job
ggplot(bank, aes(x=job, fill=factor(y))) +
  geom_bar(position="fill") +
  labs(title="Mosaic Plot of Job and Response Variable",
       x="Job",
       fill="Response (y)") +
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))
#############################################################################
#mosiac plot of response and month
ggplot(bank, aes(x=month, fill=factor(y))) +
  geom_bar(position="fill") +
  labs(title="Mosaic Plot of Month and Response Variable",
       x="Month",
       fill="Response (y)") +
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))
#############################################################################
#mosiac plot of response and poutcome
ggplot(bank, aes(x=poutcome, fill=factor(y)))+
  geom_bar(position="fill")+
  labs(title="Mosaic Plot of Poutcome and Response",
       x="poutcome",
       fill="response(y)")+
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))
#############################################################################
#heatmap of correlation matrix
numericVars <- sapply(bank, is.numeric)
corMatrix <- cor(bank[, numericVars])
corrplot(corMatrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45)
#############################################################################
#grouped box plot of duration and response
ggplot(bank, aes(x=factor(y), y=duration)) +
  geom_boxplot() +
  labs(title="Box Plot of Duration by Response Variable",
       x="Response (y)",
       y="Duration") +
  theme_minimal()
#############################################################################
#############################################################################
###########################################################
#create a training set from the data
set.seed(123)
train<-sample(1:nrow(bank), size=0.8*nrow(bank))
x<-model.matrix(y~., data=bank)[, -1]
#############################################################################
#Fit a KNN with K chosen optimally using training error rate. Report both the training and test MSE
#rates for the optimal K.
xTrain <- scale(x[train, ])
trainMean <- attr(xTrain, "scaled:center")
trainSD   <- attr(xTrain, "scaled:scale")
yTrain <- bank$y[train]
xTest <- scale(x[-train, ], center = trainMean, scale = trainSD)
yTest  <- bank$y[-train]

#initializing vector for training errors
knnTrainError <- numeric(20)
knnTestError <- numeric(20)

#loop for the KNN errors for 20 iterations
for(i in 1:20){
  knnTrain <- knn(xTrain, xTrain, yTrain, k = i)
  knnTrainError[i]<- mean(knnTrain != yTrain)
}
for(i in 1:20){
  knnTest <- knn(xTrain, xTest, yTrain, k = i)
  knnTestError[i]<- mean(knnTest != yTest)
}
#plotting the training and test error rates for KNN to determine the minimum error rate.
size=range(c(knnTrainError, knnTestError))
plot(1:20, knnTrainError, type="l", col="red", ylim=size, xlab="K", ylab="Error Rate")
lines(1:20, knnTestError, type="l", col="black")
legend("topright", legend=c("Train", "Test"), col=c("red", "black"), lty=1)
#k=13 looks like a favorable K value
optimalK <- 13
optimalK

#optimal k is 13, KNN values and confusion matrix for k=13
knnPredTrain <- knn(train = xTrain, test = xTrain, cl = yTrain, k = optimalK)
trainingMSE  <- mean(knnPredTrain != yTrain)

knnPredTest <- knn(train = xTrain, test = xTest, cl = yTrain, k = optimalK)
testMSE     <- mean(knnPredTest != yTest)

trainingMSE
testMSE
summary(knnPredTest)
#confusion matrix and error rate for KNN with K=1
knnConfusionMatrix<-table(Predicted = knnPredTest, Actual = yTest)
knnConfusionMatrix
####Confusion matrix gg plot
knnCmData <- as.data.frame(table(Predicted = knnPredTest, Actual = yTest))

# Plot using ggplot 
ggplot(knnCmData, aes(x = factor(Actual), y = factor(Predicted), fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), fontface = "bold", size = 6) +
  scale_fill_gradient(low = "#e0f3f8", high = "#084594") +
  scale_x_discrete(limits = c("0", "1"), position = "top") + 
  scale_y_discrete(limits = c("1", "0")) + 
  labs(title = "Confusion Matrix: KNN (Test Set)",
       subtitle = paste("Test Error Rate:", round(testMSE, 4)),
       x = "Actual",
       y = "Predicted") +
  theme_minimal() +
  theme(panel.grid = element_blank())
#######
#sensitivity and specificity for KNN with K=13
knnSensitivity <- knnConfusionMatrix[2, 2] / sum(knnConfusionMatrix[, 2])
knnSpecificity <- knnConfusionMatrix[1, 1] / sum(knnConfusionMatrix[, 1])
cat("KNN Sensitivity:", round(knnSensitivity, 4), "\n")
cat("KNN Specificity:", round(knnSpecificity, 4), "\n")

#decision boundary plots
plot_cols <- c("duration", "age")
xTrain_2d <- scale(bank[train, plot_cols])
# Scale parameters from training to use for the grid
train_mean <- attr(xTrain_2d, "scaled:center")
train_sd   <- attr(xTrain_2d, "scaled:scale")

# Create the grid
px1 <- seq(min(bank$duration), max(bank$duration), length.out=200)
py1 <- seq(min(bank$age), max(bank$age), length.out=200)
grid <- expand.grid(duration = px1, age = py1)

# Scale the grid using training data parameters
gridScaled <- scale(grid, center = train_mean, scale = train_sd)

# Predict on the grid using the 2D model
gridPred <- knn(train = xTrain_2d, test = gridScaled, cl = yTrain, k = optimalK)

# Reshape predictions into a matrix for the plot
gridMatrix <- matrix(as.numeric(gridPred), nrow=length(px1), ncol=length(py1))

# Generate the plot
image(px1, py1, gridMatrix, col = c("#FF000033", "#0000FF33"), 
      xlab = "Duration", ylab = "Age", 
      main = paste("KNN Decision Boundary (K =", optimalK, ")"))
# Add actual training points
points(bank$duration[train], bank$age[train], 
       col = ifelse(yTrain == "yes", "blue", "red"), pch = 20, cex = 0.5)

#############################################################################
#Now report test MSE using Random Forest with B = 500, and B = 1000. Compare your findings.
rf500 <- randomForest(xTrain, as.factor(yTrain), ntree=500)
rf1000 <- randomForest(xTrain, as.factor(yTrain), ntree=1000)

rf500Pred <- predict(rf500, xTest)
rf1000Pred <- predict(rf1000, xTest)

rf500MSE <- mean(rf500Pred != as.factor(yTest))
rf1000MSE <- mean(rf1000Pred != as.factor(yTest))

rf500MSE
rf1000MSE
#confusion matricies
table(Predicted = rf500Pred, Actual = bank$y[-train])
table(Predicted = rf1000Pred, Actual = bank$y[-train])

# Calculate variable importance
varImpData <- as.data.frame(importance(rf1000))
varImpData$variableName <- rownames(varImpData)
#consolidate grouping of month, day, and duration into one variable for the plot
factor_prefixes <- c("month", "job", "education", 
                     "marital", "contact", 
                     "poutcome", "default", 
                     "housing", "loan")
#create grouped variable names
varImpData$groupedName <- varImpData$variableName
# loop through prefixes and rename
for (prefix in factor_prefixes) {
  varImpData$groupedName <- ifelse(grepl(paste0("^", prefix), varImpData$variableName), 
                                  prefix, varImpData$groupedName)
}
#aggregate data by the new grouped variable names
aggVarImp <- aggregate(MeanDecreaseGini ~ groupedName, data = varImpData, sum)
aggVarImp <- aggVarImp[order(aggVarImp$MeanDecreaseGini, decreasing = TRUE), ]

#plot aggregated variable importance
ggplot(aggVarImp, aes(x = reorder(groupedName, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Random Forest: Variable Importance (Aggregated)",
       subtitle = "Which feature groups contribute most to predicting a 'Yes'?",
       x = "Bank Marketing Feature Groups",
       y = "Mean Decrease in Gini") +
  theme_minimal()


###################
#Confusion matrix
rf1000CmData <- as.data.frame(table(Predicted = rf1000Pred, Actual = bank$y[-train]))

ggplot(rf1000CmData, aes(x = factor(Actual), y = factor(Predicted), fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), fontface = "bold", size = 6) +
  scale_fill_gradient(low = "#e0f3f8", high = "#084594") +
  scale_x_discrete(limits = c("0", "1"), position = "top") + 
  scale_y_discrete(limits = c("1", "0")) + 
  labs(title = "Confusion Matrix: Random Forest (B=1000)",
       subtitle = paste("Test Error Rate:", round(rf1000MSE, 4)),
       x = "Actual",
       y = "Predicted") +
  theme_minimal() +
  theme(panel.grid = element_blank())

#sensitivity and specificity for random forest with B=1000
rf1000ConfMat <- table(Predicted = rf1000Pred, Actual = bank$y[-train])
rf1000Sensitivity <- rf1000ConfMat[2, 2] / sum(rf1000ConfMat[, 2])
rf1000Specificity <- rf1000ConfMat[1, 1] / sum(rf1000ConfMat[, 1])
cat("Random Forest (B=1000) Sensitivity:", round(rf1000Sensitivity, 4), "\n")
cat("Random Forest (B=1000) Specificity:", round(rf1000Specificity, 4), "\n")

#############################################################################
#############################################################################
#repeat random forest with a Boosting approach with B = 1000, d = 1, and λ = 0.01
# Convert all character columns to factors for gbm
bank[sapply(bank, is.character)] <- lapply(bank[sapply(bank, is.character)], as.factor)
boostModel <- gbm(y~., data = bank[train, ], distribution = "bernoulli", n.trees = 1000)
boostPred <- predict(boostModel, bank[-train,], type = "response")
boostMSE <- mean((boostPred > 0.5) != yTest)
boostMSE
boostProbs <- ifelse(boostPred > 0.5, 1, 0)
table(Predicted = boostProbs, Actual = bank$y[-train])
gbm.perf(boostModel, method = "OOB")
# Capture the influence data without auto-plotting
boostImportance <- summary(boostModel, plotit = FALSE)

# Plot using ggplot for a professional look
ggplot(boostImportance, aes(x = reorder(var, rel.inf), y = rel.inf)) +
  geom_bar(stat = "identity", fill = "#084594") +
  coord_flip() +
  labs(title = "Boosting: Relative Influence of Variables",
       subtitle = "Which features contribute most to predicting a 'Yes'?",
       x = "Bank Marketing Features",
       y = "Relative Influence (%)") +
  theme_minimal()
# Convert table to a data frame for plotting
confMatDf <- as.data.frame(table(Predicted = boostProbs, Actual = bank$y[-train]))
#confusion matrix heat map
ggplot(confMatDf, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  scale_fill_gradient(low = "#d1e5f0", high = "#053061") +
  labs(title = "Boosting: Confusion Matrix Heatmap",
       subtitle = paste("Overall Test MSE:", round(boostMSE, 4)),
       x = "Actual (0 = No, 1 = Yes)",
       y = "Predicted (0 = No, 1 = Yes)") +
      scale_x_discrete(limits = c("0", "1"), position = "top") + 
      scale_y_discrete(limits = c("1", "0")) + 
  theme_minimal()
#sensitivity and specificity for boosting
boostConfMat <- table(Predicted = boostProbs, Actual = bank$y[-train])
boostSensitivity <- boostConfMat[2, 2] / sum(boostConfMat[, 2])
boostSpecificity <- boostConfMat[1, 1] / sum(boostConfMat[, 1])
cat("Boosting Sensitivity:", round(boostSensitivity, 4), "\n")
cat("Boosting Specificity:", round(boostSpecificity, 4), "\n")
# To get the test error for every tree
boostTestError <- predict(boostModel, bank[-train, ], n.trees = 1:1000, type = "response")

# Calculate MSE for each stage
testMseStage <- apply(boostTestError, 2, function(pred) mean((pred > 0.5) != yTest))
trainMseStage <- boostModel$train.error # gbm stores training error automatically

# Plot both lines
plotData <- data.frame(
  trees = 1:1000,
  trainError = trainMseStage,
  testError = testMseStage
)

ggplot(plotData, aes(x = trees)) +
  geom_line(aes(y = trainError, color = "Training Error"), size = 1) +
  geom_line(aes(y = testError, color = "Test Error"), size = 1) +
  labs(title = "Boosting: Training vs. Test Error Rate",
       x = "Number of Trees",
       y = "MSE (Error Rate)",
       color = "Dataset") +
  theme_minimal()

library(pROC)

# Generate the ROC curve for your Boosting
rocObj <- roc(yTest, boostPred)

# Plot the ROC Curve
plot(rocObj, col = "blue", main = paste("ROC Curve for Boosting (AUC =", round(auc(rocObj), 3), ")"))
abline(a = 0, b = 1, lty = 2, col = "red") # Random chance line
#############################################################################

########################SVM Approach########################################

#############################################################################
#Repeat (c) with an SVM approach based on a radial basis function (choose any 𝛾) and a dth degree
#polynomial kernel (use d = 2 and d = 3). 
#radial basis function
svmRadial <- svm(as.factor(y)~ ., data = bank[train, ], kernel = "radial", gamma = 0.098)
svmRadialPred <- predict(svmRadial, newdata = bank[-train, ])
svmRadialError <- mean(svmRadialPred != yTest)
svmRadialError
table(Predicted = svmRadialPred, Actual = bank$y[-train])

#polynomial kernel with d=2
svmPoly2 <- svm(as.factor(y) ~ ., data = bank[train, ], kernel = "polynomial", degree = 2)
svmPoly2Pred <- predict(svmPoly2, newdata = bank[-train, ])
svmPoly2Error <- mean(svmPoly2Pred != yTest)
svmPoly2Error
table(Predicted = svmPoly2Pred, Actual = bank$y[-train])

#polynomial kernel with d=3
svmPoly3 <- svm(as.factor(y) ~ ., data = bank[train, ], kernel = "polynomial", degree = 3)
svmPoly3Pred <- predict(svmPoly3, newdata = bank[-train, ])
svmPoly3Error <- mean(svmPoly3Pred != yTest)
svmPoly3Error
table(Predicted = svmPoly3Pred, Actual = bank$y[-train])
###########################################################################################
#Radial COnfusion Matrix
svmCmData <- as.data.frame(table(Predicted = svmRadialPred, Actual = bank$y[-train]))

ggplot(svmCmData, aes(x = factor(Actual), y = factor(Predicted), fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), fontface = "bold", size = 6) +
  scale_fill_gradient(low = "#f7fbff", high = "#08306b") +
  scale_x_discrete(limits = c("0", "1"), position = "top") + 
  scale_y_discrete(limits = c("1", "0")) + 
  labs(title = "Confusion Matrix: SVM (Radial)",
       subtitle = paste("Test Error Rate:", round(svmRadialError, 4)),
       x = "Actual",
       y = "Predicted") +
  theme_minimal() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())
#sensitivity and specificity for radial SVM
svmRadialConfMat <- table(Predicted = svmRadialPred, Actual = bank$y[-train])
svmRadialSensitivity <- svmRadialConfMat[2, 2] / sum(svmRadialConfMat[, 2])
svmRadialSpecificity <- svmRadialConfMat[1, 1] / sum(svmRadialConfMat[, 1])
cat("Radial SVM Sensitivity:", round(svmRadialSensitivity, 4), "\n")
cat("Radial SVM Specificity:", round(svmRadialSpecificity, 4), "\n")

##Polynomial d=2 Confusion Matrix
svmPoly2CmData <- as.data.frame(table(Predicted = svmPoly2Pred, Actual = bank$y[-train]))
ggplot(svmPoly2CmData, aes(x = factor(Actual), y = factor(Predicted), fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), fontface = "bold", size = 6) +
  scale_fill_gradient(low = "#f7fbff", high = "#08306b") +
  # Force Actual 0 -> 1 (Left to Right)
  scale_x_discrete(limits = c("0", "1"), position = "top") + 
  # Force Predicted 0 -> 1 (Top to Bottom)
  scale_y_discrete(limits = c("1", "0")) + 
  labs(title = "Confusion Matrix: SVM (Polynomial d=2)",
       subtitle = paste("Test Error Rate:", round(svmPoly2Error, 4)),
       x = "Actual",
       y = "Predicted") +
  theme_minimal() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())
#sensitivity and specificity for polynomial d=2
svmPoly2ConfMat <- table(Predicted = svmPoly2Pred, Actual = bank$y[-train])
svmPoly2Sensitivity <- svmPoly2ConfMat[2, 2] / sum(svmPoly2ConfMat[, 2])
svmPoly2Specificity <- svmPoly2ConfMat[1, 1] / sum(svmPoly2ConfMat[, 1])
cat("Polynomial d=2 Sensitivity:", round(svmPoly2Sensitivity, 4), "\n")
cat("Polynomial d=2 Specificity:", round(svmPoly2Specificity, 4), "\n")

#Polynomial d=3 Confusion Matrix
svmPoly3CmData <- as.data.frame(table(Predicted = svmPoly3Pred, Actual = bank$y[-train]))
ggplot(svmPoly3CmData, aes(x = factor(Actual), y = factor(Predicted), fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), fontface = "bold", size = 6) +
  scale_fill_gradient(low = "#f7fbff", high = "#08306b") +
  scale_x_discrete(limits = c("0", "1"), position = "top") + 
  scale_y_discrete(limits = c("1", "0")) + 
  labs(title = "Confusion Matrix: SVM (Polynomial d=3)",
       subtitle = paste("Test Error Rate:", round(svmPoly3Error, 4)),
       x = "Actual",
       y = "Predicted") +
  theme_minimal() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())
#sensitivity and specificity for polynomial d=3
svmPoly3ConfMat <- table(Predicted = svmPoly3Pred, Actual = bank$y[-train])
svmPoly3Sensitivity <- svmPoly3ConfMat[2, 2] / sum(svmPoly3ConfMat[, 2])
svmPoly3Specificity <- svmPoly3ConfMat[1, 1] / sum(svmPoly3ConfMat[, 1])
cat("Polynomial d=3 Sensitivity:", round(svmPoly3Sensitivity, 4), "\n")
cat("Polynomial d=3 Specificity:", round(svmPoly3Specificity, 4), "\n")
###########################################################################################
# Create a small data frame of your SVM results
svmResults <- data.frame(
  kernelType = c("Radial", "Poly d=2", "Poly d=3"),
  testMse = c(svmRadialError, svmPoly2Error, svmPoly3Error)
)

ggplot(svmResults, aes(x = kernelType, y = testMse, fill = kernelType)) +
  geom_bar(stat = "identity") +
  labs(title = "SVM Performance by Kernel Type",
       x = "Kernel",
       y = "Test MSE") +
  theme_minimal() +
  guides(fill = "none")
# Plotting the decision boundary for the Radial SVM, duration vs age
plot(svmRadial, bank[train, ], duration ~ age, 
     color.palette = terrain.colors,
     svSymbol = 16,
     dataSymbol = NA)

# Polynomial2 Decision Boundary, duration vs age
plot(svmPoly2, bank[train, ], duration ~ age, 
     color.palette = topo.colors,
     svSymbol = 16,
     dataSymbol = NA)

# Polynomial3 Decision Boundary, duration vs age
plot(svmPoly3, bank[train, ], duration ~ age, 
     color.palette = topo.colors,
     svSymbol = 16,
     dataSymbol = NA)

#############################################################################
# ROC graph, first we retrain the Radial model with probability = TRUE
radialModel <- svm(as.factor(y) ~ ., data = bank[train, ], 
                   kernel = "radial", gamma = 0.098, probability = TRUE)
poly2Model  <- svm(as.factor(y) ~ ., data = bank[train, ], 
                   kernel = "polynomial", degree = 2, probability = TRUE)
poly3Model  <- svm(as.factor(y) ~ ., data = bank[train, ], 
                   kernel = "polynomial", degree = 3, probability = TRUE)

# predict with probability = TRUE
radialPred <- predict(radialModel, newdata = bank[-train, ], probability = TRUE)
poly2Pred  <- predict(poly2Model,  newdata = bank[-train, ], probability = TRUE)
poly3Pred  <- predict(poly3Model,  newdata = bank[-train, ], probability = TRUE)

#grab the yes probabilites for roc graph
radialProbs <- attr(radialPred, "probabilities")[, "1"]
poly2Probs  <- attr(poly2Pred, "probabilities")[, "1"]
poly3Probs  <- attr(poly3Pred, "probabilities")[, "1"]

# Create and plot the ROC Curve
radialRoc <- roc(yTest, radialProbs) 
poly2Roc  <- roc(yTest, poly2Probs)
poly3Roc  <- roc(yTest, poly3Probs)

plot(radialRoc, col = "darkred", lwd = 3, 
     main = paste("ROC Comparison for SVM Kernels"))
lines(poly2Roc, col = "darkblue", lwd = 3)
lines(poly3Roc, col = "darkgreen", lwd = 3)
legend("bottomright", legend = c(paste("Radial (AUC =", round(auc(radialRoc), 3), ")"),
                               paste("Poly d=2 (AUC =", round(auc(poly2Roc), 3), ")"),
                               paste("Poly d=3 (AUC =", round(auc(poly3Roc), 3), ")")),
       col = c("darkred", "darkblue", "darkgreen"), lwd = 3)
abline(a = 0, b = 1, lty = 2, col = "gray")
#############################################################################

#########################           Neural net         ##################################

################################################################################################

training<-sample(1:nrow(bank), size=0.8*nrow(bank))

#scaling model matrix
x<- scale(model.matrix(y~. -1, data=bank))
#force numerics
x<- matrix(as.numeric(x), nrow=nrow(x))
y<- as.numeric(bank$y)

xTrainNN<-x[training,]
yTrainNN<-y[training]

xTestNN<-x[-training,]
yTestNN<-y[-training]

#build model
model <- keras_model_sequential()

model$add(layer_input(shape = ncol(xTrainNN)))
model$add(layer_dense(units = 10, activation = "relu"))
model$add(layer_dropout(rate = 0.3))
model$add(layer_dense(units = 1, activation = "sigmoid"))

model|>compile(loss= "binary_crossentropy",
               optimizer = optimizer_rmsprop(),
               metrics = list("accuracy"))

history <- model|>fit(xTrainNN, yTrainNN,
                      epochs = 20,
                      batch_size = 32,
                      validation_data = list(xTestNN, yTestNN))

plot(history)

# Create ROC object
nnProbs <- model |> predict(xTestNN)

# Convert to binary classes (0 or 1)
nnPreds <- ifelse(nnProbs > 0.5, 1, 0)

# Calculate Test MSE (Error Rate)
nnTestMse <- mean(nnPreds != yTestNN)
cat("Neural Network Test MSE:", nnTestMse)
nnRoc <- roc(yTestNN, as.numeric(nnProbs))

# Plot ROC
plot(nnRoc, col = "darkorchid", lwd = 3, 
     main = paste("Neural Network ROC Curve (AUC =", round(auc(nnRoc), 3), ")"))
abline(a = 0, b = 1, lty = 2, col = "gray") # Random guess line

#confusion matrix table for neural net
nnProbs <- model |> predict(xTestNN)
nnPreds <- ifelse(nnProbs > 0.5, 1, 0)
table(Predicted = nnPreds, Actual = yTestNN)

#COnfusion Matrix for Neural Net
nnCmData <- as.data.frame(table(Predicted = nnPreds, Actual = yTestNN))

ggplot(nnCmData, aes(x = factor(Actual), y = factor(Predicted), fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), fontface = "bold", size = 6) +
  scale_fill_gradient(low = "#efedf5", high = "#756bb1") +
  scale_x_discrete(limits = c("0", "1"), position = "top") + 
  scale_y_discrete(limits = c("1", "0")) + 
  labs(title = "Confusion Matrix: Neural Network",
       subtitle = paste("Test Error Rate:", round(nnTestMse, 4)),
       x = "Actual",
       y = "Predicted") +
  theme_minimal() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())

##sensitivity and specificity for neural net
nnConfMat <- table(Predicted = nnPreds, Actual = yTestNN)
nnSensitivity <- nnConfMat[2, 2] / sum(nnConfMat[, 2])
nnSpecificity <- nnConfMat[1, 1] / sum(nnConfMat[, 1])
cat("Neural Network Sensitivity:", round(nnSensitivity, 4), "\n")
cat("Neural Network Specificity:", round(nnSpecificity, 4), "\n")

################################################################################
# Consolidate results into one table
summary_results <- data.frame(
  Method = c("KNN", "Random Forest", "Boosting", "SVM (Radial)", "SVM(Poly2)", "SVM(Poly3)","Neural Net"),
  ErrorRate = c(testMSE, rf1000MSE, boostMSE, svmRadialError, svmPoly2Error, svmPoly3Error, nnTestMse)
)

# Plot the comparisons of error rates
ggplot(summary_results, aes(x = reorder(Method, ErrorRate), y = ErrorRate, fill = Method)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(ErrorRate, 4)), vjust = -0.5, fontface = "bold") +
  labs(title = "Final Comparison: Which Model Performed Best?",
       subtitle = "Lower Error Rate is better",
       x = "Model Type", y = "Test Error Rate (MSE)") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set2") +
  guides(fill = "none")

#comparison plot of sensitivty and specificity for the best models
sensitivity_specificity <- data.frame(
  Model = c("Random Forest", "Boosting", "SVM (Radial)", "SVM(Poly2)", "SVM(Poly3)", "Neural Net", "KNN"),
  Sensitivity = c(rf1000Sensitivity, boostSensitivity, svmRadialSensitivity, svmPoly2Sensitivity, svmPoly3Sensitivity, nnSensitivity, knnSensitivity),
  Specificity = c(rf1000Specificity, boostSpecificity, svmRadialSpecificity, svmPoly2Specificity, svmPoly3Specificity, nnSpecificity, knnSpecificity)
)
ggplot(sensitivity_specificity, aes(x = Model)) +
  geom_bar(aes(y = Sensitivity), stat = "identity", alpha = 0.7) +
  geom_bar(aes(y = Specificity), stat = "identity", fill = "salmon", alpha = 0.7) +
  geom_text(aes(y = Sensitivity, label = round(Sensitivity, 4)), vjust = -0.5, fontface = "bold") +
  geom_text(aes(y = Specificity, label = round(Specificity, 4)), vjust = -0.5, fontface = "bold") +
  labs(title = "Specificity and Sensitivity Comparison",
       subtitle = "Higher is better for both metrics",
       x = "Model Type", y = "Value") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set1") +
  guides(fill = "none") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


##############################################################################################
#Stepwise Model selection stuff
bankModel1<-glm(y~1, data=bank, family="binomial")
bankModel2<-glm(y~., data=bank, family="binomial")
bankModel2
summary(bankModel2)

vif(bankModel2)

forwardAIC<-step(bankModel1, scope=list(lower=bankModel1, upper=bankModel2), 
                 direction="forward", k=2)
forwardAIC
summary(forwardAIC)

stepwiseAIC<-step(bankModel1, scope=list(lower=bankModel1, upper=bankModel2), 
                  direction="both", k=2)
stepwiseAIC
summary(stepwiseAIC)

backwardAIC<-step(bankModel2, scope=list(lower=bankModel2, upper=bankModel1), 
                  direction="backward", k=2)
backwardAIC
summary(backwardAIC)

AIC(forwardAIC)
AIC(stepwiseAIC)
AIC(backwardAIC)

plot(forwardAIC$fitted.values, forwardAIC$residuals,
     xlab = "Fitted Values",
     ylab = "Residuals",
     main = "Fitted vs. Residuals")
abline(h = 0, col = "red", lwd=3)
qqnorm(forwardAIC$residuals)
qqline(forwardAIC$residuals, col="red", lwd=2)

group1<-forwardAIC$fitted.values>median(forwardAIC$fitted.values)
var.test(forwardAIC$residuals[group1], forwardAIC$residuals[!group1])

