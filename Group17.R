# Student Performance Analysis Using Supervised Learning - STAT5009 Project
# Models: Logistic Regression, LASSO, KNN, Random Forest
# Group: 17
# -----------------------------------------------


# Load necessary libraries
library(caret)          # Model training, CV, evaluation
library(pROC)           # ROC and AUC calculations
library(glmnet)         # LASSO logistic regression
library(randomForest)   # Random Forest model
library(ggplot2)        # Visualization

# Step 1: Load data
data <- read.table("student-por.csv", sep = ";", header = TRUE)

# --------------------------
# EXPLORATORY DATA ANALYSIS
# --------------------------

# Summary of dataset structure
str(data)

# Summary statistics of numeric variables
summary(data)

# Histogram of final grade G3
ggplot(data, aes(x = G3)) + 
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Final Grade (G3)", x = "Final Grade (G3)", y = "Count")

# Bar plot of Pass/Fail distribution
data$G3_class <- factor(ifelse(data$G3 >= 10, "Pass", "Fail"))
ggplot(data, aes(x = G3_class, fill = G3_class)) +
  geom_bar() +
  labs(title = "Count of Pass vs Fail", x = "Class", y = "Count")

# Boxplot of absences by pass/fail
ggplot(data, aes(x = G3_class, y = absences, fill = G3_class)) +
  geom_boxplot() +
  labs(title = "Absences by Pass/Fail", x = "Class", y = "Number of Absences")

# Bar Chart of Pass/Fail by Guardian
ggplot(data, aes(x = guardian, fill = G3_class)) +
  geom_bar(position = "dodge") +
  labs(title = "Pass/Fail Distribution by Guardian Type",
       x = "Guardian", y = "Count")


# --------------------------
# DATA PREPROCESSING
# --------------------------

# Remove G1 and G2 to simulate early prediction
data <- data[, !(names(data) %in% c("G1", "G2"))]

# Convert character columns to factor
data[sapply(data, is.character)] <- lapply(data[sapply(data, is.character)], as.factor)

# Recreate G3_class factor (in case removal changed it)
data$G3_class <- factor(ifelse(data$G3 >= 10, "Pass", "Fail"))

# --------------------------
# TRAIN-TEST SPLIT (80/20)
# --------------------------

index <- createDataPartition(data$G3_class, p = 0.8, list = FALSE)
train <- data[index, ]
test <- data[-index, ]

# --------------------------
# MODELING SETUP
# --------------------------

# Control setup for caret repeated CV with ROC metric
ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3,
                     classProbs = TRUE, summaryFunction = twoClassSummary)

# --------------------------
# 1. LOGISTIC REGRESSION
# --------------------------

# Train logistic regression on selected predictors
logit_model <- train(G3_class ~ failures + higher + school + absences,
                     data = train,
                     method = "glm",
                     family = binomial,
                     trControl = ctrl,
                     metric = "ROC")

print(logit_model)

# Predict probabilities and classes on test
logit_probs <- predict(logit_model, test, type = "prob")[, "Pass"]
logit_preds <- predict(logit_model, test)

# Confusion matrix and accuracy
cat("Logistic Regression Confusion Matrix:\n")
print(confusionMatrix(logit_preds, test$G3_class))

# ROC curve and AUC
roc_logit <- roc(test$G3_class, logit_probs)
plot(roc_logit, main = "ROC Curve - Logistic Regression", print.auc = TRUE)

# --------------------------
# 2. LASSO LOGISTIC REGRESSION
# --------------------------

# Prepare matrices for glmnet
x_train <- model.matrix(G3_class ~ failures + higher + school + absences, train)[, -1]
y_train <- ifelse(train$G3_class == "Pass", 1, 0)

x_test <- model.matrix(G3_class ~ failures + higher + school + absences, test)[, -1]
y_test <- ifelse(test$G3_class == "Pass", 1, 0)

# 10-fold CV to find best lambda for LASSO
cv_lasso <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1, type.measure = "deviance")
plot(cv_lasso)
title("LASSO Logistic Regression - CV Deviance")

best_lambda <- cv_lasso$lambda.min
cat("Best lambda (LASSO):", best_lambda, "\n")

# Predict on test at best lambda
lasso_probs <- predict(cv_lasso, newx = x_test, s = best_lambda, type = "response")
lasso_preds <- factor(ifelse(lasso_probs >= 0.5, "Pass", "Fail"), levels = c("Fail", "Pass"))

cat("LASSO Logistic Regression Confusion Matrix:\n")
print(confusionMatrix(lasso_preds, test$G3_class))

roc_lasso <- roc(test$G3_class, as.vector(lasso_probs))
plot(roc_lasso, main = "ROC Curve - LASSO Logistic Regression", print.auc = TRUE)

# --------------------------
# 3. K-NEAREST NEIGHBORS (KNN)
# --------------------------

# Select predictors and convert categorical to dummy
knn_train <- train[, c("failures", "higher", "school", "absences")]
knn_test <- test[, c("failures", "higher", "school", "absences")]

dummies <- dummyVars(~ ., data = knn_train)
train_knn_x <- predict(dummies, knn_train)
test_knn_x <- predict(dummies, knn_test)

# Scale features (mean=0, sd=1)
train_knn_x <- scale(train_knn_x)
test_knn_x <- scale(test_knn_x)

train_knn_y <- train$G3_class

# Train KNN with caret, tune k using CV
knn_fit <- train(x = train_knn_x,
                 y = train_knn_y,
                 method = "knn",
                 tuneLength = 10,
                 trControl = ctrl,
                 metric = "ROC")

print(knn_fit)

# Predict on test data
knn_preds <- predict(knn_fit, test_knn_x)
knn_probs <- predict(knn_fit, test_knn_x, type = "prob")[, "Pass"]

cat("KNN Confusion Matrix:\n")
print(confusionMatrix(knn_preds, test$G3_class))

roc_knn <- roc(test$G3_class, knn_probs)
plot(roc_knn, main = "ROC Curve - KNN", print.auc = TRUE)

# --------------------------
# 4. RANDOM FOREST
# --------------------------

rf_fit <- train(G3_class ~ . -G3,
                data = train,
                method = "rf",
                ntree = 500,
                tuneLength = 5,
                trControl = ctrl,
                metric = "ROC")

print(rf_fit)

rf_preds <- predict(rf_fit, test)
rf_probs <- predict(rf_fit, test, type = "prob")[, "Pass"]

cat("Random Forest Confusion Matrix:\n")
print(confusionMatrix(rf_preds, test$G3_class))

roc_rf <- roc(test$G3_class, rf_probs)
plot(roc_rf, main = "ROC Curve - Random Forest", print.auc = TRUE)

# --------------------------
# MODEL PERFORMANCE SUMMARY
# --------------------------

results <- data.frame(
  Model = c("Logistic Regression", "LASSO Logistic Regression", "KNN", "Random Forest"),
  Accuracy = c(
    mean(logit_preds == test$G3_class),
    mean(lasso_preds == test$G3_class),
    mean(knn_preds == test$G3_class),
    mean(rf_preds == test$G3_class)
  ),
  AUC = c(
    auc(roc_logit),
    auc(roc_lasso),
    auc(roc_knn),
    auc(roc_rf)
  )
)

cat("\nModel Performance Summary:\n")
print(results)

