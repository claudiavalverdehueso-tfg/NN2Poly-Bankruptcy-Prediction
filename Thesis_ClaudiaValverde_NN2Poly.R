# APPLICATION OF THE INTERPRETABILITY TECHNIQUE: NN2POLY 
# ============================================================

# This script applies the NN2Poly methodology to the trained neural network.
# The objective is to reformulate the neural network into an equivalent symbolic
# polynomial representation, enhancing interpretability while preserving predictive capacity.


# 1. Environment Setup
# ------------------------------------------------------------
# Configure Python environment and load required libraries 
# for neural network training and NN2Poly transformation.

rm(list=ls())

Sys.setenv(RETICULATE_USE_MANAGED_VENV = "no")
Sys.setenv(RETICULATE_PYTHON = "/Users/claudia/opt/anaconda3/envs/r-nn2poly/bin/python")

library(reticulate)
py_config()
py_run_string("import tensorflow as tf, keras; print(tf.__version__); print(keras.__version__)")

# Load libraries required for modeling and polynomial reformulation
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


# 2. Data Preparation
# ------------------------------------------------------------
# Load the preprocessed dataset and reproduce the same
# train-test framework used in the modeling stage.
# The training set is balanced, while the test set remains
# unbalanced to preserve real-world prevalence.

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


# 3. Neural Network Construction (NN2Poly Compatible)
# ------------------------------------------------------------
# Build neural network using the optimal hyperparameters
# identified in the neural network modeling notebook.

# The output layer uses linear activation (logits)
# because NN2Poly operates on logits.
# The sigmoid transformation will be applied separately when
# converting logits into probabilities.

nn_model <- keras_model_sequential() %>%
  layer_dense(units = 50, activation = "tanh", input_shape = ncol(x_train_balanced)) %>%
  layer_dense(units = 1, activation = "linear")


# 4. Regularization for Interpretability
# ------------------------------------------------------------
# Apply L1 norm constraints to encourage sparsity in the
# resulting polynomial representation.
# This reduces the magnitude of less relevant coefficients,
# improving interpretability of the symbolic model.

nn_model <- nn2poly::add_constraints(object = nn_model, type = "l1_norm")


# 5. Model Training
# ------------------------------------------------------------
# The training dataset is manually split into training and
# validation subsets to ensure controlled threshold selection.

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
# Binary cross-entropy with logits is used since the output
# layer is linear and does not apply sigmoid internally.
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


# 6. Threshold Optimization 
# ------------------------------------------------------------
# Since the model outputs probabilities after sigmoid transformation,
# the optimal classification threshold is selected using Youden's J statistic.
# This balances sensitivity and specificity.

thresholds <- seq(0.01, 0.99, by = 0.01)

# Function computes sensitivity, specificity, accuracy, and Youden's J 
# for a given probability threshold.
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

# Identify threshold maximizing Youden's J statistic
best_row <- df_thr[which.max(df_thr$youdenJ), ]
best_thr <- best_row$threshold
best_row
best_thr


# 7. Neural Network Test Evaluation
# ------------------------------------------------------------
# Evaluate the trained neural network on the held-out test set
# using the optimized threshold derived from validation data.

probability_model <- keras_model(
  inputs = nn_model$input,
  outputs = nn_model$output %>% layer_activation("sigmoid")
)

prediction_NN_class <- predict(probability_model, x_test)

# Convert predicted logits into class predictions using optimal threshold
prediction_NN_class <- ifelse(prediction_NN_class > best_thr, 1, 0) 

prediction_NN <- predict(nn_model, x_test)  

# Show confusion matrix
cm_nn <- caret::confusionMatrix(as.factor(prediction_NN_class), as.factor(y_test))
cm_nn


# 8. NN2Poly Reformulation
# ------------------------------------------------------------
# Reformulate the trained neural network into an equivalent
# symbolic polynomial representation up to order 3.
# This transformation preserves the functional mapping
# of the neural network while exposing its internal structure.

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


# 9. Polynomial Approximation Validation
# ------------------------------------------------------------
# Compare neural network logits with polynomial logits
# to verify that the symbolic approximation faithfully
# reproduces the neural network predictions.

# Confusion matrix: NN classes vs Poly classes (agreement between both)
cm_2 <- caret::confusionMatrix(as.factor(prediction_NN_class), as.factor(prediction_poly_class))
cm_2

# Diagonal plot: agreement between NN logits and polynomial logits
nn2poly:::plot_diagonal(
  x_axis = prediction_NN,
  y_axis = prediction_poly,
  xlab = "NN Prediction (logits)",
  ylab = "Polynomial prediction (logits)"
)

# High agreement between logits confirms that the polynomial
# accurately approximates the neural network's internal mapping.


# 10. Taylor Expansion Diagnostics (NN2Poly)
# ------------------------------------------------------------
# NN2Poly relies on a Taylor expansion of activation functions to obtain
# a polynomial representation. The following plots show:
# - the Taylor approximation quality, and
# - activation potential diagnostics,
# for both the hidden layer and the output layer.

# A well-behaved approximation ensures that higher-order terms
# meaningfully capture nonlinear interactions among financial ratios.

# Prepare training data in the format expected by nn2poly diagnostic functions
train <- data.frame(x_train_balanced, target = y_train_balanced)

plots <- nn2poly:::plot_taylor_and_activation_potentials(
  object = nn_model,
  data = train,
  max_order = 3,
  constraints = TRUE
)

print(plots[[1]])  # hidden layer diagnostics
print(plots[[2]])  # output layer diagnostics


# 11. Polynomial Term Importance
# ------------------------------------------------------------
# This section examines the resulting polynomial representation and
# inspects the most influential polynomial terms.

# Top 10 most significant polynomial terms (by magnitude)
# These terms represent the strongest contributors in the symbolic model.
plot(poly_model, n = 10)


# 12. Interpretation of Polynomial Coefficients
# ------------------------------------------------------------
# Extract polynomial terms and associated coefficients.
# Coefficients with larger absolute magnitude indicate greater influence
# on the model output (logit), and therefore stronger impact on bankruptcy risk.

# Convert NN2Poly term labels into readable term strings
terms <- sapply(poly_model$labels, function(x) paste(x, collapse = ","))

# Extract coefficients (associated with each polynomial term)
coefs <- poly_model$values[, 1]

coefs_df <- data.frame(
  Term = terms,
  Coefficient = coefs,
  Abs_Coef = abs(coefs)
)

# Translate polynomial index terms into financial ratio names
traducir_termino <- function(termino) {
  indices <- as.numeric(unlist(strsplit(termino, ",")))
  nombres <- column_names[indices]
  paste(nombres, collapse = " * ")
}

coefs_df$Variable <- sapply(coefs_df$Term, traducir_termino)

# Rank polynomial terms by absolute coefficient magnitude
top_vars <- coefs_df[order(-coefs_df$Abs_Coef), ]

# Display the 10 most influential terms
head(top_vars[, c("Variable", "Coefficient")], 10)

# Visualization of top polynomial terms
ggplot(head(top_vars, 10), aes(x = reorder(Variable, Abs_Coef), y = Coefficient)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Top 10 most Influential Polynomial Terms",
       x = "Variables or interactions",
       y = "Coefficients") +
  theme_minimal()

# The symbolic polynomial representation reveals the most influential
# financial ratios and their nonlinear interactions driving bankruptcy risk.
# Unlike the original neural network, this formulation provides
# transparent and economically interpretable terms.