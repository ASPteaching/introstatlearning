library(keras)       # for modeling
library(tidyverse)   # for wrangling & visualization
library(glue)        # for string literals

mnist <- dataset_mnist()

# reshape 3D tensor (aka array) to a 2D tensor (aka matrix)
train_images <- array_reshape(mnist$train$x, c(60000, 28 * 28))
test_images <- array_reshape(mnist$test$x, c(10000, 28 * 28))

# standardize train and test features
train_images <- train_images / 255
test_images <- test_images / 255

# get number of features
n_feat <- ncol(train_images)

# Convert labels to categorical
train_labels <- to_categorical(mnist$train$y, num_classes = 10)
test_labels <- to_categorical(mnist$test$y, num_classes = 10)

# Define model architecture
model <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = 'relu', input_shape = c(n_feat)) %>%
  layer_dense(units = 10, activation = 'softmax')

# Define how the model is going to learn
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "sgd",
  metrics = "accuracy"
)

# Summary of the model
summary(model)

# Train the model
history <- model %>% fit(
  train_images, train_labels,
  validation_split = 0.2,
  batch_size = 32,
  epochs = 10
)
