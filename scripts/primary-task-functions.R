library(tidyverse)
library(tidytext)
library(textstem)
library(rvest)
library(qdapRegex)
library(Matrix)
 
parse_pages <- function(.df) {
  clean <- function(.html) {
    tryCatch(read_html(.html) %>%
               html_elements("title, h1, h2, h3, h4, h5, h6, p") %>%
               html_text2() %>%
               str_c(collapse = " "),
             error = function(e) "") %>%
      rm_url() %>%
      rm_email() %>%
      str_remove_all("'") %>%
      str_replace_all("\n|[[:punct:]]|nbsp|[[:digit:]]|[[:symbol:]]", " ") %>%
      str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
      tolower()
  }
  .df %>% mutate(text_clean = map_chr(text_tmp, clean))
}
 

tokenize_pages <- function(.df) {
  .df %>%
    unnest_tokens(token, text_clean) %>%
    filter(!token %in% stop_words$word) %>%
    mutate(term = lemmatize_words(token)) %>%
    filter(str_length(term) > 2) %>%
    select(.id, term)
}
 

fit_featurizer <- function(tokens) {
  df <- tokens %>% distinct(.id, term) %>% count(term) %>% filter(n >= 3)
  list(vocab = df$term, idf = log(n_distinct(tokens$.id) / df$n))
}
 

make_dtm <- function(featurizer, tokens, ids) {
  counts <- tokens %>%
    filter(term %in% featurizer$vocab) %>%
    count(.id, term)
  x <- sparseMatrix(i = match(counts$.id, ids),
                    j = match(counts$term, featurizer$vocab),
                    x = log1p(counts$n),
                    dims = c(length(ids), length(featurizer$vocab)))
  x <- x %*% Diagonal(x = featurizer$idf)
  norms <- pmax(sqrt(rowSums(x^2)), 1e-8)
  as(Diagonal(x = 1 / norms) %*% x, "RsparseMatrix")
}