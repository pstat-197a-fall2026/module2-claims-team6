library(tidyverse)
library(tidymodels)
library(modelr)
library(Matrix)
library(sparsesvd)
library(glmnet)


source('scripts/preprocessing.R')


load('data/claims-raw.RData')

claims_clean <- claims_raw %>%
  parse_data() %>% 
  nlp_fn()

save(claims_clean, file = 'data/claims-clean-with-headers.RData')

load('data/claims-clean-with-headers.RData')

set.seed(15131)
claims_split <- claims_clean %>%
  initial_split(prop = 0.8)

train_dtm <- training(claims_split) %>% select(-.id, -bclass)
train_labels <- training(claims_split) %>% select(.id, bclass)

test_dtm <- testing(claims_split) %>% select(-.id, -bclass)
test_labels <- testing(claims_split) %>% select(.id, bclass)

url <- 'https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/activities/data/'

source(paste(url, 'projection-functions.R', sep = ''))

proj_out <- projection_fn(.dtm = train_dtm, .prop = 0.7) 

train_dtm_projected <- proj_out$data

proj_out$n_pc

train <- train_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(train_dtm_projected)

fit <- glm(bclass ~ ., data = train, family = "binomial")

test_dtm_projected <- reproject_fn(.dtm = test_dtm, proj_out)

preds <- predict(fit, newdata = test_dtm_projected, type = "response")

pred_df <- test_labels %>% transmute(bclass = factor(bclass))  %>% 
  bind_cols(pred = as.numeric(preds)) %>% 
  mutate(bclass.pred = factor(pred > 0.5, labels = levels(bclass)))

panel <- metric_set(sensitivity, specificity, accuracy, roc_auc)



load('data/claims-clean-example.RData')
claims_clean_paragraph <- claims_clean %>% nlp_fn()

set.seed(15131)
claims_split_paragraph <- claims_clean_paragraph %>%
  initial_split(prop = 0.8)

train_dtm_par <- training(claims_split_paragraph) %>% select(-.id, -bclass)
train_labels_par <- training(claims_split_paragraph) %>% select(.id, bclass)

test_dtm_par <- testing(claims_split_paragraph) %>% select(-.id, -bclass)
test_labels_par <- testing(claims_split_paragraph) %>% select(.id, bclass)

proj_out_par <- projection_fn(.dtm = train_dtm_par, .prop = 0.7) 

train_dtm_projected_par <- proj_out_par$data

proj_out_par$n_pc

train_par<- train_labels_par %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(train_dtm_projected_par)

fit_par <- glm(bclass ~ ., data = train_par, family = "binomial")

test_dtm_projected_par <- reproject_fn(.dtm = test_dtm_par, proj_out_par)

preds_par <- predict(fit_par, newdata = test_dtm_projected_par, type = "response")

pred_df_par <- test_labels_par %>% transmute(bclass = factor(bclass))  %>% 
  bind_cols(pred = as.numeric(preds_par)) %>% 
  mutate(bclass.pred = factor(pred > 0.5, labels = levels(bclass)))

panel_par <- metric_set(sensitivity, specificity, accuracy, roc_auc)

pred_df %>% panel(truth = bclass, estimate = bclass.pred, pred, event_level = 'second')

pred_df_par %>% panel_par(truth = bclass, estimate = bclass.pred, pred, event_level = 'second')

#Including header information helped to slightly improve the roc_auc from 0.841 to 0.873

