# Training an image recognizer on MNIST data

# Install & libraries
# devtools::install_github("rstudio/keras")

library(keras)

# set hyperparameter flags

FLAGS <- flags(
  flag_numeric("hl1", 256),
  flag_numeric("hl2", 128)
)

# input layer: use MNIST images
mnist <- dataset_mnist()
x_train <- mnist$train$x; y_train <- mnist$train$y
x_test <- mnist$test$x; y_test <- mnist$test$y


# reshape and rescale
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
x_train <- x_train / 255; x_test <- x_test / 255
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# defining the model and layers
model <- keras_model_sequential()
model %>%
layer_dense(units = FLAGS$hl1, activation = 'relu', input_shape = c(784)) %>%
layer_dense(units = FLAGS$hl2, activation = 'relu') %>%
layer_dense(units = 10, activation = 'softmax')

summary(model)

# compile (define loss and optimizer)
model %>% compile(loss = 'categorical_crossentropy',
                  optimizer = optimizer_rmsprop(),
                  metrics = c('accuracy'))
# train (fit)
model %>% fit(x_train, y_train, epochs = 20, 
              batch_size = 128, validation_split = 0.2)


#evaluate
score <- model %>% evaluate(
  x_test, y_test,
  verbose = 0
)

cat('Test loss:', score[1], '\n')
cat('Test accuracy:', score[2], '\n')

