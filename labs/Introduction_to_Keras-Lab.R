
knitr::opts_chunk$set(echo = TRUE, message = FALSE, warning = FALSE)
ggplot2::theme_set(ggplot2::theme_minimal())



knitr::include_graphics("images/kerasPipeline.png")



library(keras)       # for modeling
library(tidyverse)   # for wrangling & visualization
library(glue)        # for string literals



mnist <- dataset_mnist()
str(mnist)



# 60K images of 28x28 pixels
dim(mnist$train$x)

# pixel values are gray scale ranging from 0-255
range(mnist$train$x)



digit <- mnist$train$x[1,,]
digit



plot(as.raster(digit, max = 255))



par(mfrow = c(10, 10), mar = c(0,0,0,0))
for (i in 1:100) {
  plot(as.raster(mnist$train$x[i,,], max = 255))
}



c(c(train_images, train_labels), c(test_images, test_labels)) %<-% mnist

# the above is the same as
# train_images <- mnist$train$x
# train_labels <- mnist$train$y
# test_images <- mnist$test$x
# test_labels <- mnist$test$y



m <- matrix(1:9, nrow = 3)
m

# flattened matrix
as.vector(m)

# Warning of how you invert the process !!!
matrix(as.vector(m), nrow=3)



# reshape 3D tensor (aka array) to a 2D tensor (aka matrix)
train_images <- array_reshape(train_images, c(60000, 28 * 28))
test_images <- array_reshape(test_images, c(10000, 28 * 28))

# our training data is now a matrix with 60K observations and
# 784 features (28 pixels x 28 pixels = 784)
str(train_images)



train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

head(train_labels)



# all our features (pixels) range from 0-255
range(train_images)



# standardize train and test features
train_images <- train_images / 255
test_images <- test_images / 255



obs <- nrow(train_images)
set.seed(123)
randomize <- sample(seq_len(obs), size = obs, replace = FALSE)
train_images <- train_images[randomize, ]
train_labels <- train_labels[randomize, ]



# get number of features
n_feat <- ncol(train_images)

# 1. Define model architecture
model <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = 'relu', input_shape = n_feat) %>%
  layer_dense(units = 10, activation = 'softmax')

# 2. Define how our model is going to learn
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "sgd",
  metrics = "accuracy"
)

summary(model)



# 3. Train our model
history <- model %>% fit(
  train_images, train_labels,
  validation_split = 0.2
  )



history



plot(history)



# define model architecture
model <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = 'relu', input_shape = n_feat) %>%
  layer_dense(units = 10, activation = 'softmax')

# define how the model is gonig to learn
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "sgd",
  metrics = "accuracy"
  )

history <- model %>% fit(
  train_images, train_labels,
  validation_split = 0.2,
  ____ = ____
  )



model <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = 'relu', input_shape = n_feat) %>%
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = ____,
  metrics = "accuracy"
  )

history <- model %>% fit(
  train_images, train_labels,
  validation_split = 0.2,
  batch_size = 128
  )



model <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = 'relu', input_shape = n_feat) %>%
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_sgd(lr = 0.1, momentum = 0.9),
  metrics = "accuracy"
)

history <- model %>% fit(
  train_images, train_labels,
  validation_split = 0.2,
  batch_size = 128,
  epochs = 20,
  callback = callback_early_stopping(patience = 3, restore_best_weights = TRUE,
                                     min_delta = 0.0001)
  )



history



plot(history)



model <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = 'relu', input_shape = n_feat) %>%
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_sgd(lr = 0.1, momentum = 0.9),
  metrics = "accuracy"
)

history <- model %>% fit(
  train_images, train_labels,
  validation_split = 0.2,
  batch_size = 128,
  epochs = 20,
  callback = list(
    callback_early_stopping(patience = 3, restore_best_weights = TRUE, min_delta = 0.0001),
    callback_reduce_lr_on_plateau(patience = 1, factor = 0.1)
    )
  )



history



plot(history)



plot(history$metrics$lr)



model <- keras_model_sequential() %>%
  ___________ %>%
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_sgd(lr = 0.1, momentum = 0.9),
  metrics = "accuracy"
)

history <- model %>% fit(
  train_images, train_labels,
  validation_split = 0.2,
  batch_size = 128,
  epochs = 20,
  callback = list(
    callback_early_stopping(patience = 3, restore_best_weights = TRUE, min_delta = 0.0001),
    callback_reduce_lr_on_plateau(patience = 1, factor = 0.1)
    )
  )



train_model <- function(n_units, n_layers, log_to) {
  
  # Create a model with a single hidden input layer
  model <- keras_model_sequential() %>%
    layer_dense(units = n_units, activation = "relu", input_shape = n_feat)
  
  # Add additional hidden layers based on input
  if (n_layers > 1) {
    for (i in seq_along(n_layers - 1)) {
      model %>% layer_dense(units = n_units, activation = "relu")
    }
  }
  
  # Add final output layer
  model %>% layer_dense(units = 10, activation = "softmax")
  
  # compile model
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_sgd(lr = 0.1, momentum = 0.9),
    metrics = "accuracy"
  )
  
  # train model and store results with callback_tensorboard()
  history <- model %>% fit(
    train_images, train_labels,
    validation_split = 0.2,
    batch_size = 128,
    epochs = 20,
    callback = list(
      callback_early_stopping(patience = 3, restore_best_weights = TRUE, min_delta = 0.0001),
      callback_reduce_lr_on_plateau(patience = 1, factor = 0.1),
      callback_tensorboard(log_dir = log_to)
      ),
    verbose = FALSE
    )
  
  return(history)
  }



grid <- expand_grid(
  units = c(128, 256, 512, 1024),
  layers = c(1:3)
) %>%
  mutate(id = paste0("mlp_", layers, "_layers_", units, "_units"))
grid



for (row in seq_len(nrow(grid))) {
  # get parameters
  units <- grid[[row, "units"]]
  layers <- grid[[row, "layers"]]
  file_path <- paste0("mnist/", grid[[row, "id"]])
  
  # provide status update
  cat(layers, "hidden layer(s) with", units, "neurons: ")
  
  # train model
  m <- train_model(n_units = units, n_layers = layers, log_to = file_path)
  min_loss <- min(m$metrics$val_loss, na.rm = TRUE)
  
  # update status with loss
  cat(min_loss, "\n", append = TRUE)
}



tensorboard("mnist")



model <- keras_model_sequential() %>%
  ___________ %>%
  ___________

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_sgd(lr = 0.1, momentum = 0.9),
  metrics = "accuracy"
)

history <- model %>% fit(
  train_images, train_labels,
  validation_split = 0.2,
  batch_size = 128,
  epochs = 20,
  callback = list(
    callback______(patience = _____, restore_best_weights = TRUE, min_delta = 0.0001),
    callback______(patience = _____, factor = 0.1)
    )
  )



model <- keras_model_sequential() %>%
  layer_dense(
    units = 512, activation = "relu", input_shape = n_feat,
    kernel_regularizer = regularizer_l2(l = 0.001)    # regularization parameter
    ) %>%
  layer_dense(
    units = 512, activation = "relu",
    kernel_regularizer = regularizer_l2(l = 0.001)    # regularization parameter
    ) %>%
  layer_dense(units = 10, activation = "softmax")

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_sgd(lr = 0.1, momentum = 0.9),
  metrics = "accuracy"
)

history <- model %>% fit(
  train_images, train_labels,
  validation_split = 0.2,
  batch_size = 128,
  epochs = 20,
  callback = list(
    callback_early_stopping(patience = 3, restore_best_weights = TRUE, min_delta = 0.0001),
    callback_reduce_lr_on_plateau(patience = 1, factor = 0.1)
    )
  )



history



plot(history)



model <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", input_shape = n_feat) %>%
  layer_dropout(0.3) %>%                            # regularization parameter
  layer_dense(units = 512, activation = "relu") %>%
  layer_dropout(0.3) %>%                           # regularization parameter
  layer_dense(units = 10, activation = "softmax")

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_sgd(lr = 0.1, momentum = 0.9),
  metrics = "accuracy"
)

history <- model %>% fit(
  train_images, train_labels,
  validation_split = 0.2,
  batch_size = 128,
  epochs = 20,
  callback = list(
    callback_early_stopping(patience = 3, restore_best_weights = TRUE, min_delta = 0.0001),
    callback_reduce_lr_on_plateau(patience = 1, factor = 0.1)
    )
  )



history



plot(history)



model %>% evaluate(test_images, test_labels, verbose = FALSE)



predictions <- model %>% predict_classes(test_images)
actual <- mnist$test$y



missed_predictions <- sum(predictions != actual)
missed_predictions



caret::confusionMatrix(factor(predictions), factor(actual))



tibble(
  actual,
  predictions
  ) %>% 
  filter(actual != predictions) %>%
  count(actual, predictions) %>%
  mutate(perc = n / n() * 100) %>% 
  filter(n > 1) %>% 
  ggplot(aes(actual, predictions, size = n)) +
  geom_point(shape = 15, col = "#9F92C6") +
  scale_x_continuous("Actual Target", breaks = 0:9) +
  scale_y_continuous("Prediction", breaks = 0:9) +
  scale_size_area(breaks = c(2, 5, 10, 15), max_size = 5) +
  coord_fixed() +
  ggtitle(paste(missed_predictions, "mismatches")) +
  theme(panel.grid.minor = element_blank()) +
  labs(caption = 'Adapted from Rick Scavetta')



missed <- which(predictions != actual)
plot_dim <- ceiling(sqrt(length(missed)))

par(mfrow = c(plot_dim, plot_dim), mar = c(0,0,0,0))
for (i in missed) {
  plot(as.raster(mnist$test$x[i,,], max = 255))
}



par(mfrow = c(4, 4), mar = c(0,0,2,0))

for (i in missed[1:16]) {
  plot(as.raster(mnist$test$x[i,,], max = 255)) 
  title(main = paste("Predicted:", predictions[i]))
}

