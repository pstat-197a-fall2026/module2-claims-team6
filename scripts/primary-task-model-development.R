source("scripts/primary-task-functions.R")
library(tidymodels)
library(LiblineaR)
 
load("data/claims-raw.RData")
claims <- claims_raw %>% select(.id, bclass, mclass, text_tmp) %>% parse_pages()
 

set.seed(197)
split <- initial_split(claims, prop = 0.8, strata = mclass)
train <- training(split)
test  <- testing(split)
 
tok_train <- tokenize_pages(train)
fz <- fit_featurizer(tok_train)
x_train <- make_dtm(fz, tok_train, train$.id)
x_test  <- make_dtm(fz, tokenize_pages(test), test$.id)
 

set.seed(1)
folds <- sample(rep(1:5, length.out = nrow(train)))
cv_acc <- function(y, cost) {
  mean(sapply(1:5, function(k) {
    fit <- LiblineaR(x_train[folds != k, ], y[folds != k], type = 2, cost = cost)
    mean(predict(fit, x_train[folds == k, ])$predictions == y[folds == k])
  }))
}
costs <- c(0.1, 0.3, 1, 3, 10)
best_cost <- sapply(c("bclass", "mclass"), function(t)
  costs[which.max(sapply(costs, cv_acc, y = train[[t]]))])
best_cost
 

fit_b <- LiblineaR(x_train, train$bclass, type = 2, cost = best_cost["bclass"])
fit_m <- LiblineaR(x_train, train$mclass, type = 2, cost = best_cost["mclass"])
holdout <- test %>%
  mutate(bclass.pred = factor(predict(fit_b, x_test)$predictions, levels(bclass)),
         mclass.pred = factor(predict(fit_m, x_test)$predictions, levels(mclass)))
 
panel <- metric_set(sensitivity, specificity, accuracy)
holdout %>% panel(truth = bclass, estimate = bclass.pred, event_level = "second")
holdout %>% panel(truth = mclass, estimate = mclass.pred, estimator = "macro")
 

tok_all <- tokenize_pages(claims)
fz_all <- fit_featurizer(tok_all)
x_all <- make_dtm(fz_all, tok_all, claims$.id)
 
for (t in c("bclass", "mclass")) {
  model <- list(fit = LiblineaR(x_all, claims[[t]], type = 2, cost = best_cost[t]),
                featurizer = fz_all,
                levels = levels(claims[[t]]))
  saveRDS(model, paste0("results/", t, "-svm-model.rds"))
}