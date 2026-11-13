source("scripts/primary-task-functions.R")
library(LiblineaR)
 
load("data/claims-test.RData")
test <- parse_pages(claims_test)
tokens <- tokenize_pages(test)
 
predict_model <- function(path) {
  model <- readRDS(path)
  x <- make_dtm(model$featurizer, tokens, test$.id)
  factor(predict(model$fit, x)$predictions, levels = model$levels)
}
 
pred_df <- tibble(.id = test$.id,
                  bclass.pred = predict_model("results/bclass-svm-model.rds"),
                  mclass.pred = predict_model("results/mclass-svm-model.rds"))
 
save(pred_df, file = "results/preds-group6.RData")