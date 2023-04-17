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
text(tree, pretty = 0)

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
  pruned_tree <- prune.misclass(tree, best = 5)
}

# Compare the training error rates between the pruned and unpruned trees
train_preds <- predict(tree, OJ_train, type = "class")
train_error_unpruned <- mean(train_preds != OJ_train$Purchase)
cat("Training error rate for unpruned tree:", train_error_unpruned, "\n")

train_preds <- predict(pruned_tree, OJ_train, type = "class")
train_error_pruned <- mean(train_preds != OJ_train$Purchase)
cat("Training error rate for pruned tree:", train_error_pruned, "\n")
