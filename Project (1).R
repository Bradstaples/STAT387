library(class)

# 1. Load with factors
bank <- read.csv("bank_marketing.csv", sep = ";", stringsAsFactors = TRUE)
for(i in bank){
  bank$y == "yes", 1, 0)
}


bankModel<-glm(y~., data=bank, family="binomial")
summary(bankModel)
#create a training set from the data
train<-sample(1:nrow(bank), size=0.8*nrow(bank))

#create matricies for pre processing
x<-as.matrix(y~., data=bankModel)



# 2. Pre-process for mobank# 2. Pre-process for models needing numeric data (KNN/SVM)
# This creates dummy variables for all categorical columns
X_matrix <- model.matrix(y ~ . - 1, data = bank)
X_scaled <- scale(X_matrix)
y_numeric <- ifelse(bank$y == "yes", 1, 0)

# 3. Split the data (80/20)
set.seed(123)
train_idx <- sample(1:nrow(bank), size = 0.8 * nrow(bank))

# For KNN/SVM
train_X <- X_scaled[train_idx, ]
test_X  <- X_scaled[-train_idx, ]
train_y <- y_numeric[train_idx]
test_y  <- y_numeric[-train_idx]

# For Trees (Random Forest/Boosting)
train_df <- bank[train_idx, ]
test_df  <- bank[-train_idx, ]

# --- PART B: KNN ---
# Note: K=1 will always have 0% training error. Mention this in your paper!
k_values <- 1:20
train_errs <- sapply(k_values, function(k) {
  pred <- knn(train_X, train_X, train_y, k = k)
  mean(pred != train_y)
})
optimal_k <- k_values[which.min(train_errs)] # Likely K=1

# --- PART D: Boosting (Fixed) ---
# Ensure y is numeric 0/1 for gbm
train_df_boost <- train_df
train_df_boost$y <- ifelse(train_df_boost$y == "yes", 1, 0)

boost_model <- gbm(y ~ ., data = train_df_boost, 
                   distribution = "bernoulli", 
                   n.trees = 1000, 
                   interaction.depth = 1, 
                   shrinkage = 0.01)

# Prediction returns probabilities; must convert to classes for Error Rate
boost_prob <- predict(boost_model, newdata = test_df, n.trees = 1000, type = "response")
boost_pred <- ifelse(boost_prob > 0.5, "yes", "no")
boost_error <- mean(boost_pred != test_df$y)

