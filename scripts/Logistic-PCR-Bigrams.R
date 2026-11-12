# Part 1: 

library(tidyverse)
library(tidymodels)
library(tidytext)
library(textstem)
library(Matrix)
library(sparsesvd)
library(glmnet)
library(modelr)

source("scripts/preprocessing.R")
load("data/claims-raw.RData")

claims_parsed <- claims_raw %>%
  parse_data()

glimpse(claims_parsed)

dim(claims_parsed)


# Part 2: Create Word-tokenized DTM

claims_word <- claims_parsed %>%
  nlp_fn()

dim(claims_word)



# Part 3: Create Bigram-tokenized DTM

bigram_fn <- function(.df) {
  
  out <- .df %>%
    unnest_tokens(
      output = bigram,
      input = text_clean,
      token = "ngrams",
      n = 2
    ) %>%
    separate(
      bigram,
      into = c("word1", "word2"),
      sep = " "
    ) %>%
    filter(
      !word1 %in% stop_words$word,
      !word2 %in% stop_words$word
    ) %>%
    mutate(
      word1 = lemmatize_words(word1),
      word2 = lemmatize_words(word2)
    ) %>%
    filter(
      str_length(word1) > 2,
      str_length(word2) > 2
    ) %>%
    mutate(
      bigram = str_c(word1, word2, sep = "_")
    ) %>%
    count(.id, bclass, bigram, name = "n") %>%
    bind_tf_idf(
      term = bigram,
      document = .id,
      n = n
    ) %>%
    pivot_wider(
      id_cols = c(".id", "bclass"),
      names_from = bigram,
      values_from = tf_idf,
      values_fill = 0
    )
  
  return(out)
}


claims_bigram <- bigram_fn(claims_parsed)

dim(claims_bigram)



# Part 4: Create a common training/testing split

common_ids <- intersect(
  claims_word$.id,
  claims_bigram$.id
)

claims_labels <- claims_parsed %>%
  distinct(.id, bclass) %>%
  filter(.id %in% common_ids)

nrow(claims_labels)

set.seed(15131)

claims_split <- initial_split(
  claims_labels,
  prop = 0.8,
  strata = bclass
)

train_ids <- training(claims_split)$.id
test_ids <- testing(claims_split)$.id

length(train_ids)
length(test_ids)



# Part 5: Extract training and testing DTM

get_dtm <- function(.dtm, .ids) {
  .dtm %>%
    filter(.id %in% .ids) %>%
    arrange(match(.id, .ids))
}


word_train <- get_dtm(claims_word, train_ids)
word_test <- get_dtm(claims_word, test_ids)


bigram_train <- get_dtm(claims_bigram, train_ids)
bigram_test <- get_dtm(claims_bigram, test_ids)


dim(word_train)
dim(word_test)

dim(bigram_train)
dim(bigram_test)


# Part 6: Check binary class distribution

train_labels <- training(claims_split) %>%
  select(.id, bclass)

train_labels %>%
  count(bclass)

test_labels <- testing(claims_split) %>%
  select(.id, bclass)

test_labels %>%
  count(bclass)

levels(train_labels$bclass)



# Part 7: Load PCA projection functions

url <- paste0(
  "https://raw.githubusercontent.com/",
  "pstat197/pstat197a/main/",
  "materials/activities/data/"
)

source(paste0(url, "projection-functions.R"))

print(projection_fn)

print(reproject_fn)



# Part 8: Word Logistic PCR

train_dtm <- word_train %>%
  select(-.id, -bclass)

proj_word <- projection_fn(
  .dtm = train_dtm,
  .prop = 0.7
)

proj_word$n_pc


# Use the training data's PCA scores
word_train_pc <- proj_word$data

word_test_pc <- reproject_fn(
  .dtm = word_test %>% select(-.id, -bclass),
  .projection_fn_out = proj_word
)

dim(word_train_pc)
dim(word_test_pc)


# Identify Bigram feature columns
bigram_feature_names <- setdiff(
  names(bigram_train),
  c(".id", "bclass")
)

# Calculate the number of documents containing each Bigram
bigram_doc_freq <- vapply(
  bigram_train[bigram_feature_names],
  function(x) sum(x != 0),
  numeric(1)
)

# Select the 2,000 most frequent Bigrams
bigram_top_features <- names(
  sort(bigram_doc_freq, decreasing = TRUE)
)[1:2000]

# Keep only the selected Bigram features
bigram_train_small <- bigram_train %>%
  select(all_of(bigram_top_features))

bigram_test_small <- bigram_test %>%
  select(all_of(bigram_top_features))

# Check the dimensions
dim(bigram_train_small)
dim(bigram_test_small)


# Perform PCA on the selected Bigram training data
proj_bigram <- projection_fn(
  .dtm = bigram_train_small,
  .prop = 0.7
)
# Check the number of principal components
proj_bigram$n_pc   


# Use the training data's Bigram PCA scores
bigram_train_pc <- proj_bigram$data
# Project the test data using the same PCA loadings
bigram_test_pc <- reproject_fn(
  .dtm = bigram_test_small,
  .projection_fn_out = proj_bigram
)
# Check the dimensions of the projected datasets
dim(bigram_train_pc)
dim(bigram_test_pc)



######################### First-stage Word Logistic PCR #########################

# Prepare the predictor matrix and binary response variable
word_x <- as.matrix(word_train_pc)

word_y <- as.integer(
  train_labels$bclass == "Relevant claim content"
)



# Set a random seed for reproducibility
set.seed(15131)

# Perform 5-fold cross-validation to select the ridge penalty
# Fit a ridge-regularized logistic PCR model using Word principal components.
word_ridge_cv <- cv.glmnet(
  x = word_x,
  y = word_y,
  family = "binomial",
  alpha = 0,
  nfolds = 5,
  type.measure = "deviance"
)

# Display the selected regularization parameter
word_ridge_cv$lambda.min


# Generate Word model predictions for the training data
word_train_logodds <- as.numeric(
  predict(
    word_ridge_cv,
    newx = word_x,
    s = "lambda.min",
    type = "link"
  )
)

# Generate Word model predictions for the test data
word_test_logodds <- as.numeric(
  predict(
    word_ridge_cv,
    newx = as.matrix(word_test_pc),
    s = "lambda.min",
    type = "link"
  )
)

# Check the distribution of predicted log-odds
summary(word_train_logodds)

# Check the range of predicted log-odds
range(word_train_logodds)

# Check for non-finite predictions
sum(!is.finite(word_train_logodds))

# Combine Word model predictions with Bigram principal components
train_combined <- bind_cols(
  tibble(word_logodds = word_train_logodds),
  bigram_train_pc
)

test_combined <- bind_cols(
  tibble(word_logodds = word_test_logodds),
  bigram_test_pc
)

# Add the training and testing class labels
train_combined <- bind_cols(
  train_labels %>% select(bclass),
  train_combined
)

test_combined <- bind_cols(
  test_labels %>% select(bclass),
  test_combined
)

# Check the dimensions of the combined datasets
dim(train_combined)
dim(test_combined)

######################### Second-stage Logistic Regression #########################

# Fit the second logistic regression model
combined_model <- glm(
  bclass ~ .,
  data = train_combined,
  family = binomial()
)

# Display the model summary
summary(combined_model)

# Predict probabilities using the combined model
combined_test_prob <- predict(
  combined_model,
  newdata = test_combined,
  type = "response"
)

# Check the distribution of predicted probabilities
summary(combined_test_prob)


# Calculate the ROC AUC for the combined model
combined_test_results <- test_labels %>%
  mutate(.pred_relevant = combined_test_prob)

combined_auc <- roc_auc(
  combined_test_results,
  truth = bclass,
  .pred_relevant,
  event_level = "second"
)

combined_auc

# Convert Word model log-odds to predicted probabilities
word_test_prob <- plogis(word_test_logodds)

# Calculate the ROC AUC for the Word-only model
word_test_results <- test_labels %>%
  mutate(.pred_relevant = word_test_prob)

word_auc <- roc_auc(
  word_test_results,
  truth = bclass,
  .pred_relevant,
  event_level = "second"
)

word_auc



# Part 9: Compare Word-only and Word + Bigram models

auc_comparison <- tibble(
  model = c("Word-only", "Word + Bigram"),
  test_auc = c(
    word_auc$.estimate,
    combined_auc$.estimate
  )
)

auc_comparison

# Calculate the change in test AUC after adding Bigrams

auc_difference <- combined_auc$.estimate - word_auc$.estimate

auc_difference


# The results suggest that bigrams do not provide substantial additional predictive information 
# about the claims status of a page. The Word-only model achieved a test AUC of 0.857, while the 
# combined model using word-tokenized predictions and bigram principal components achieved a 
# slightly lower test AUC of 0.850. The decrease of approximately 0.007 indicates that adding 
# bigram features did not improve the model's predictive performance on the test set. Therefore, 
# based on these results, there is no clear evidence that bigrams capture additional useful 
# information about claims status beyond what is already captured by individual words.




# ### Preliminary Task 2: Bigram Analysis
# 
# For this task, we performed a secondary tokenization of the page text to create bigram features. We first fitted a ridge-regularized logistic principal component regression model using the word-tokenized data, retaining principal components that explained 70% of the variance. We then obtained predicted log-odds from this model and combined them with principal components derived from the 2,000 most frequent bigram features in a second logistic regression model.
# 
# We evaluated the predictive performance of the Word-only model and the combined Word + Bigram model using test-set ROC AUC. The Word-only model achieved an AUC of 0.857, compared with 0.850 for the combined model. These results suggest that adding bigram features did not improve predictive performance, and there is no clear evidence that bigrams provide additional useful information about claims status beyond individual words in this analysis.
# 
