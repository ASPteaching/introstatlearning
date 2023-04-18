
#############################################
### Tree building and optimization
#############################################

# Using the `tree` package



# Load the ISLR2 package and the OJ data set
library(ISLR2)
data(OJ)

# Create a training set containing a random sample of 800 observations, and a test set containing the remaining observations
set.seed(1)
train_idx <- sample(nrow(OJ), 800)
OJ_train <- OJ[train_idx, ]
OJ_test <- OJ[-train_idx, ]

# Fit a tree to the training data, with Purchase as the response and the other variables as predictors
library(tree)
tree <- tree(Purchase ~ ., data = OJ_train)

# Use the summary() function to produce summary statistics about the tree, and describe the results obtained
summary(tree)
cat("Training error rate for unpruned tree:", tree$dev/sum(tree$frame$wt), "\n")
cat("Number of terminal nodes in unpruned tree:", tree$frame$var[tree$frame$var == "<leaf>"] %>% length, "\n")

# Create a plot of the tree
plot(tree)
text(tree, pretty = 0, cex=0.8)

# Predict the response on the test data, and produce a confusion matrix comparing the test labels to the predicted test labels
test_preds <- predict(tree, OJ_test, type = "class")
table(test_preds, OJ_test$Purchase)

# Calculate the test error rate
test_error_unpruned <- mean(test_preds != OJ_test$Purchase)
cat("Test error rate for unpruned tree:", test_error_unpruned, "\n")

# Apply the cv.tree() function of the tree package to the training set in order to determine the optimal tree size
cv <- cv.tree(tree, FUN = prune.misclass)

# Give the code to Produce a plot with tree size on the x-axis and cross-validated classification error rate on the y-axis
plot(cv$size, cv$dev, type = "b", xlab = "Tree Size", ylab = "Cross-Validated Error Rate")

# Use the tree package to Produce a pruned tree corresponding to the optimal tree size obtained using cross-validation
pruned_tree <- prune.misclass(tree, best = cv$size[which.min(cv$dev)])

# If cross-validation does not select a pruned tree, create a tree with five terminal nodes
if (is.null(pruned_tree)) {
  pruned_tree <- prune.misclass(tree, best = 8)
}

# Compare the training error rates between the pruned and unpruned trees
train_preds <- predict(tree, OJ_train, type = "class")
train_error_unpruned <- mean(train_preds != OJ_train$Purchase)
cat("Training error rate for unpruned tree:", train_error_unpruned, "\n")

train_preds <- predict(pruned_tree, OJ_train, type = "class")
train_error_pruned <- mean(train_preds != OJ_train$Purchase)
cat("Training error rate for pruned tree:", train_error_pruned, "\n")



# Using the `rpart` package

# load the necessary packages
library(rpart)
library(rpart.plot)

# load the OJ dataset
data(OJ)

# split the data into training and test sets
set.seed(123)
train <- sample(nrow(OJ), 800)
train_data <- OJ[train, ]
test_data <- OJ[-train, ]

# fit a decision tree to the training data
tree_model <- rpart(Purchase ~ ., data = train_data, method = "class")

# summarize the tree model
summary(tree_model)

# plot the tree model
rpart.plot(tree_model, type = 0)

# make predictions on the test data
test_preds <- predict(tree_model, test_data, type = "class")

# create a confusion matrix
conf_matrix <- table(test_data$Purchase, test_preds)

# calculate the test error rate
test_error_rate <- 1 - sum(diag(conf_matrix)) / sum(conf_matrix)

# use cross-validation to find the optimal tree size
cv_results <- rpart.control(cp = seq(0.0001, 0.01, by = 0.0001))
cv_model <- rpart(Purchase ~ ., data = train_data, method = "class", control = cv_results)
plotcp(cv_model)

# prune the tree to the optimal size
optimal_cp <- cv_model$cptable[which.min(cv_model$cptable[, "xerror"]), "CP"]
pruned_model <- prune(tree_model, cp = optimal_cp)

# compare training error rates between the pruned and unpruned trees
train_preds <- predict(tree_model, train_data, type = "class")
train_error_rate_unpruned <- 1 - sum(diag(table(train_data$Purchase, train_preds))) / nrow(train_data)

train_preds_pruned <- predict(pruned_model, train_data, type = "class")
train_error_rate_pruned <- 1 - sum(diag(table(train_data$Purchase, train_preds_pruned))) / nrow(train_data)

cat("Training error rate (unpruned):", train_error_rate_unpruned, "\n")
cat("Training error rate (pruned):", train_error_rate_pruned, "\n")

