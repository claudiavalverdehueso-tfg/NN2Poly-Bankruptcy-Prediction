# MODEL BUILDING: NN2POLY

# Clean environment
rm(list=ls())

Sys.setenv(RETICULATE_USE_MANAGED_VENV = "no")
Sys.setenv(RETICULATE_PYTHON = "/Users/claudia/opt/anaconda3/envs/r-nn2poly/bin/python")

library(reticulate)
py_config()
py_run_string("import tensorflow as tf, keras; print(tf.__version__); print(keras.__version__)")

# Load libraries
library(ggplot2)
library(cowplot)
library(caret)
library(keras)
library(nn2poly)
library(patchwork)
library(tensorflow)
library(readr) 
library(dplyr)
library(rsample)
library(ROSE)
library(pROC)

# 1) Build neural network with hyperparameters chosen

# Read data
df <- read_csv("polish_preprocessed.csv") %>%
  mutate(class = ifelse(class == 1, 1, 0)) %>%
  mutate(across(everything(), ~ifelse(is.infinite(.), NA, .))) %>%
  na.omit()

# 80/20 split (stratified)
set.seed(42)
split <- initial_split(df, prop = 0.80, strata = class)
train_df <- training(split)
test_df  <- testing(split)

# Balance train set
set.seed(123)
balanced_data <- ovun.sample(
  class ~ .,
  data = train_df,
  method = "both",
  p = 0.5,
  seed = 123
)$data

# X/Y train balanced
x_train_balanced <- as.matrix(balanced_data[, -ncol(balanced_data)])
y_train_balanced <- as.numeric(as.character(balanced_data[, ncol(balanced_data)]))

# X/Y test real (NOT balanced)
x_test <- as.matrix(select(test_df, -class))
y_test <- as.numeric(test_df$class)

# Names of variables (interpret polynomial)
colnames(x_train_balanced) <- colnames(balanced_data)[-ncol(balanced_data)]
colnames(x_test) <- colnames(x_train_balanced)
column_names <- colnames(x_train_balanced)

# Build NN
nn_model <- keras_model_sequential() %>%
  layer_dense(units = 50, activation = "tanh", input_shape = ncol(x_train_balanced)) %>%
  layer_dense(units = 1, activation = "linear")

nn_model <- nn2poly::add_constraints(object = nn_model, type = "l1_norm")

# Train NN
set.seed(45)

n <- nrow(x_train_balanced)
idx <- sample.int(n)
val_frac <- 0.2
n_val <- floor(val_frac * n)

val_idx <- idx[1:n_val]
tr_idx  <- idx[(n_val+1):n]

x_tr  <- x_train_balanced[tr_idx, , drop = FALSE]
y_tr  <- y_train_balanced[tr_idx]
x_val <- x_train_balanced[val_idx, , drop = FALSE]
y_val <- y_train_balanced[val_idx]

# Compile + fit (no validation_split already did manually)
nn_model %>% compile(
  loss = loss_binary_crossentropy(from_logits = TRUE),
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = "accuracy"
)

history <- nn_model %>% fit(
  x_tr, y_tr,
  epochs = 100,
  batch_size = 8,
  validation_data = list(x_val, y_val),
  verbose = 1
)

# Model of probabilities (sigmoid over logits)
probability_model <- keras_model(
  inputs = nn_model$input,
  outputs = nn_model$output %>% layer_activation("sigmoid")
)

# Probabilities in validation
p_val <- as.numeric(predict(probability_model, x_val))

# Thresholds
thresholds <- seq(0.01, 0.99, by = 0.01)

calc_metrics <- function(y_true, p, thr){
  y_hat <- ifelse(p >= thr, 1, 0)
  tp <- sum(y_true == 1 & y_hat == 1)
  tn <- sum(y_true == 0 & y_hat == 0)
  fp <- sum(y_true == 0 & y_hat == 1)
  fn <- sum(y_true == 1 & y_hat == 0)
  
  sens <- if ((tp + fn) == 0) NA else tp / (tp + fn)
  spec <- if ((tn + fp) == 0) NA else tn / (tn + fp)
  acc  <- (tp + tn) / (tp + tn + fp + fn)
  youden <- sens + spec - 1
  
  c(sensitivity = sens, specificity = spec, accuracy = acc, youdenJ = youden)
}

M <- t(sapply(thresholds, function(t) calc_metrics(y_val, p_val, t)))
df_thr <- data.frame(threshold = thresholds, M)

# Optimal threshold Youden J
best_row <- df_thr[which.max(df_thr$youdenJ), ]
best_thr <- best_row$threshold
best_row
best_thr

# NN predictions
probability_model <- keras_model(
  inputs = nn_model$input,
  outputs = nn_model$output %>% layer_activation("sigmoid")
)

prediction_NN_class <- predict(probability_model, x_test)

prediction_NN_class <- ifelse(prediction_NN_class > best_thr, 1, 0) 

prediction_NN <- predict(nn_model, x_test)  # logits

# Show confusion matrix
cm_nn <- caret::confusionMatrix(as.factor(prediction_NN_class), as.factor(y_test))
cm_nn


# 2) NN2Poly

# Polynomial representation
poly_model <- nn2poly(nn_model, polmax_order = 3)
poly_model

# Polynomial predictions (logits)
prediction_poly <- predict(poly_model, newdata = x_test)

sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

prediction_poly_prob <- sigmoid(prediction_poly)
prediction_poly_class <- ifelse(prediction_poly_prob > best_thr, 1, 0) 

prediction_poly_class <- factor(prediction_poly_class, levels = c(0, 1))
y_test_factor <- factor(y_test, levels = c(0, 1))

# Confusion matrix: NN classes vs Poly classes (agreement between both)
cm_2 <- caret::confusionMatrix(as.factor(prediction_NN_class), as.factor(prediction_poly_class))
cm_2

# Diagonal plot: NN logits vs Poly logits
nn2poly:::plot_diagonal(
  x_axis = prediction_NN,
  y_axis = prediction_poly,
  xlab = "NN Prediction (logits)",
  ylab = "Polynomial prediction (logits)"
)

# Top 10 most significant polynomial terms
plot(poly_model, n = 10)

# Taylor expansion performance plots
train <- data.frame(x_train_balanced, target = y_train_balanced)

plots <- nn2poly:::plot_taylor_and_activation_potentials(
  object = nn_model,
  data = train,
  max_order = 3,
  constraints = TRUE
)

print(plots[[1]])  # hidden layer
print(plots[[2]])  # output layer

# Interpretation of coefficients
terms <- sapply(poly_model$labels, function(x) paste(x, collapse = ","))

coefs <- poly_model$values[, 1]

coefs_df <- data.frame(
  Term = terms,
  Coefficient = coefs,
  Abs_Coef = abs(coefs)
)

traducir_termino <- function(termino) {
  indices <- as.numeric(unlist(strsplit(termino, ",")))
  nombres <- column_names[indices]
  paste(nombres, collapse = " * ")
}

coefs_df$Variable <- sapply(coefs_df$Term, traducir_termino)

top_vars <- coefs_df[order(-coefs_df$Abs_Coef), ]

head(top_vars[, c("Variable", "Coefficient")], 10)

ggplot(head(top_vars, 10), aes(x = reorder(Variable, Abs_Coef), y = Coefficient)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Top 10 most Influential Polynomial Terms",
       x = "Variables or interactions",
       y = "Coefficients") +
  theme_minimal()