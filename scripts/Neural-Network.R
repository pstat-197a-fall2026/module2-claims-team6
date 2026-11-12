library(tidyverse)
library(tidymodels)
library(keras)
library(tensorflow)

# Load data
source("scripts/preprocessing.R")
load("data/claims-raw.RData")

claims_parsed <- claims_raw %>%
  parse_data()

# Partition the data
set.seed(111126)
partitions <- claims_parsed %>%
  initial_split(prop = 0.8)

train_text <- training(partitions) %>%
  pull(text_clean)
train_labels <- training(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1


# function: build, compile, and train a model
run_model <- function(max_tokens = NULL,
                      ngrams = NULL,
                      output_mode = 'tf_idf',
                      units = 25,
                      dropout = 0.2,
                      l2 = 0,
                      learning_rate = 0.001,
                      epochs = 30,
                      batch_size = 32,
                      validation_split = 0.3) {
  
  tensorflow::set_random_seed(111126)
  
  # create a preprocessing layer
  preprocess_layer <- layer_text_vectorization(
    standardize = NULL,
    split = 'whitespace',
    ngrams = ngrams,
    max_tokens = max_tokens,
    output_mode = output_mode
  )
  
  preprocess_layer %>% adapt(train_text)
  
  # define NN architecture
  model <- keras_model_sequential() %>%
    preprocess_layer() %>%
    layer_dropout(dropout) %>%
    layer_dense(units = units, activation = 'relu',
                kernel_regularizer = regularizer_l2(l2)) %>%
    layer_dropout(dropout) %>%
    layer_dense(1) %>%
    layer_activation(activation = 'sigmoid')
  
  # configure for training
  model %>% compile(
    loss = 'binary_crossentropy',
    optimizer = optimizer_adam(learning_rate = learning_rate),
    metrics = list(metric_auc(name = 'auc'), 'binary_accuracy')
  )
  
  # train
  history <- model %>%
    fit(train_text,
        train_labels,
        validation_split = 0.3,
        epochs = epochs,
        batch_size = batch_size,
        callbacks = if (validation_split > 0) list(callback_early_stopping(
          monitor = 'val_loss', patience = 3, restore_best_weights = TRUE)) else NULL)
  
  return(list(model = model, history = history))
}

# Run experiments

summarize_runs <- function(runs) {
  t(sapply(runs, function(r) {
    h <- r$history$metrics
    best <- which.min(h$val_loss)
    c(epoch = best, val_loss = h$val_loss[best], val_auc = h$val_auc[best])
  }))
}

# Experimenting with dropout only (fixing overfitting)
runs <- list()

runs[["drop0.2"]] <- run_model(max_tokens = 6000, dropout = 0.2)
runs[["drop0.4"]] <- run_model(max_tokens = 6000, dropout = 0.4)
runs[["drop0.6"]] <- run_model(max_tokens = 6000, dropout = 0.6)

summarize_runs(runs)

# Pick dropout = 0.6 (val_loss = 0.5998493, val_auc = 0.8241940)

# Experimenting with hidden layer size only (how much model memorizes)
runs <- list()

runs[["drop0.6_u8"]]  <- run_model(max_tokens = 6000, dropout = 0.6, units = 8)
runs[["drop0.6_u25"]] <- run_model(max_tokens = 6000, dropout = 0.6, units = 25)
runs[["drop0.6_u64"]] <- run_model(max_tokens = 6000, dropout = 0.6, units = 64)

summarize_runs(runs)

# Pick units = 64 (val_loss = 0.6172434, val_auc = 0.8588427)

# Experiment with l2 only (overfitting)
runs <- list()

runs[["best_l2_0"]]     <- run_model(max_tokens = 6000, dropout = 0.6, units = 64, l2 = 0)
runs[["best_l2_0.001"]] <- run_model(max_tokens = 6000, dropout = 0.6, units = 64, l2 = 0.001)
runs[["best_l2_0.01"]]  <- run_model(max_tokens = 6000, dropout = 0.6, units = 64, l2 = 0.01)

summarize_runs(runs)

# Pick l2 = 0.001 (val_loss = 0.6690199, val_auc = 0.8609717)

# Experiment with bigrams and other variables
runs <- list()

runs[["base"]]       <- run_model(max_tokens = 6000, dropout = 0.6, units = 64, l2 = 0.001)
runs[["bigrams"]]    <- run_model(max_tokens = 6000, dropout = 0.6, units = 64, l2 = 0.001, ngrams = 2)
runs[["bigrams10k"]] <- run_model(max_tokens = 10000, dropout = 0.6, units = 64, l2 = 0.001, ngrams = 2)
runs[["count"]]      <- run_model(max_tokens = 6000, dropout = 0.6, units = 64, l2 = 0.001, output_mode = 'count')

summarize_runs(runs)

# Keep the same (base)

# Experiment with learning rate only
runs <- list()

runs[["lr0.001"]]  <- run_model(max_tokens = 6000, dropout = 0.6, units = 64, l2 = 0.001, learning_rate = 0.001)
runs[["lr0.0003"]] <- run_model(max_tokens = 6000, dropout = 0.6, units = 64, l2 = 0.001, learning_rate = 0.0003)
runs[["lr0.0001"]] <- run_model(max_tokens = 6000, dropout = 0.6, units = 64, l2 = 0.001, learning_rate = 0.0001)

summarize_runs(runs)

# Keep the same (lr0.001)

# Final model
final_model <- run_model(max_tokens = 6000, dropout = 0.6, units = 64, l2 = 0.001,
                 validation_split = 0)

test_text   <- testing(partitions) %>% pull(text_clean)
test_labels <- testing(partitions) %>% pull(bclass) %>% as.numeric() - 1

probs <- final_model$model %>% predict(test_text)
pROC::auc(test_labels, as.numeric(probs))

# Confidence intervals
probs <- final_model$model %>% predict(test_text)
pROC::auc(test_labels, as.numeric(probs))
pROC::ci.auc(test_labels, as.numeric(probs))

preds <- as.numeric(probs > 0.5)
mean(preds == test_labels)                    # accuracy
table(pred = preds, truth = test_labels)


## Results

# 95% CI: 0.7966-0.8746 (DeLong)
# Accuracy: 0.7850467
# truth
# pred   0   1
# 0 153  49
# 1  43 183

# The final neural network with one hidden layer, max_tokens = 6000, dropout = 0.6, units = 64, l2 = 0.001)
# achieved AUC of around 0.8356 with a 95% confidence interval of 0.7966-0.8746 and test accuracy of 0.7850467.
# It performed below Task 1, which might be due to uncertainty and data splits. Bigrams and learning rates didn't
# significantly improve the model.


## Terms

# max_tokens: N most frequent features
# ngrams: single words vs. word pairs (bigrams)
# units: number of neurons in the hidden layer (more capacity => overfitting)
# dropout: fraction of neurons randomly switched off  during training (higher values => relies less on one feature => less overfitting)
# l2: penalty on larger weights (higher values => simpler models => less overfitting)
# learning_rate:size of weight updates (lower => slower and steadier)
# batch_size: how many documents per update (smaller batches => add noise and some regularization)
# epochs: max number of passes through training data

