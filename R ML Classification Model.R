


# Loading libraries 
  

library(ggplot2) 
library(crayon)
library(dplyr) 
library(tidyverse)
library(caret) 
library(nnet)         # for multinomial logistic regression 
library(e1071)        # for confusionMatrix 
library(MASS)         # for stepAIC
library(MLmetrics)
library(reshape2)     # for melting the data frame
library(pROC)         # for ROC curve




# Loading Dataset "heart_disease.csv" 

# Current working directory
curr_work_dir <- getwd()
cat("Current working directory:", curr_work_dir, "\n") 


# defining path
new_path <- "C:/Users/shahs/..........."

# New working directory
new_dir <- setwd(new_path)
cat("New working directory:", new_dir, "\n")


# CSV file name 
file_name <- "heart_disease.csv" 

# File path
file_path <- file.path(new_dir, file_name)

# Reading CSV file
df <- read.csv(file_path)



cat("========= Display Dataset ==========\n")
rows <- head(df)
cat("First few rows of the dataset:\n")
print(rows)



# Class

cat(black$bold("========= Dataset class ==========\n"))
cat("\n")
class_df <- class(df)
print(class_df)



# Dimension 
cat(black$bold("========= Data frame dimension ==========\n"))
cat("\n") 
dim_df <- dim(df) 
print(dim_df)



# Structure 
cat(black$bold("========= Data frame structure  ==========\n"))
cat("\n") 
str_df <- str(df) 
print(str_df)


# Summary Statistics 
cat(black$bold("========= Data frame statistics ==========\n"))
cat("\n") 
basic_stats <- summary(df) 
print(basic_stats)


# Data types 
cat(black$bold("========= Data frame data types ==========\n"))
cat("\n") 
data_types <- sapply(df,class) 
print(data_types)


# Missing values 
cat(black$bold("========= Total missing values ==========\n"))
cat("\n") 
sum(is.na(df))

cat("\n\n")
cat(black$bold("========= Missing values in each column ==========\n"))
cat("\n") 
colSums(is.na(df))



# Handling missing values
cat(black$bold("========= Handling missing values ==========\n"))
cat("\n") 
my_data <- na.omit(df)  
sum(is.na(my_data))

cat("\n\n")
cat(black$bold("========= Missing values after handling ==========\n"))
cat("\n") 
colSums(is.na(my_data))   



# Character columns to factors
my_data[] <- lapply(my_data, function(col) {
  if (is.character(col)) as.factor(col) else col
})


# Numeric columns
numeric_cols <- names(my_data)[sapply(my_data, is.numeric)]

# Categorical columns 
categorical_cols <- names(my_data)[sapply(my_data, is.factor)]


# Printing numeric columns
cat("\n\n")
cat(black$bold("============= Numeric Columns: ================\n"))
cat("\n") 
print(numeric_cols)


# Printing categorical columns and their levels
cat("\n\n")
cat(black$bold("============= Categorical Columns and Their Levels: ================\n"))
cat("\n")
for (col in categorical_cols) {
  cat("\nColumn:", col, "\n")
  print(levels(my_data[[col]]))
}



################################################################################




# Chi-Square Tests for Categorical Predictors



target_var <- "Stress.Level"  

predictors <- names(my_data)[sapply(my_data, is.factor) | sapply(my_data, is.character)]
predictors <- setdiff(predictors, target_var)


for (pred in predictors) {
  cat("\n====== Chi-Square Test between", pred, "and", target_var, "=====\n")
  
  
  # Creating contingency table
  tbl <- table(my_data[[pred]], my_data[[target_var]])
  
  if (any(tbl == 0)) {
    cat("Warning: Zero counts in contingency table, test may be invalid.\n")
  }
  
  test_result <- chisq.test(tbl)
  print(test_result)
}



# Initializing results list
chi_results <- data.frame(Predictor = character(), P_Value = numeric())


for (pred in predictors) {
  tbl <- table(my_data[[pred]], my_data[[target_var]])
  test_result <- chisq.test(tbl)
  chi_results <- rbind(chi_results, data.frame(Predictor = pred, P_Value = test_result$p.value))
}

# Viewing summary
cat(black$bold("========= Chi-Squared Results Summary: ==========\n"))
cat("\n") 
print(chi_results)




############################################################################




# ANOVA with Numeric Predictors 



# Numeric predictors
numeric_predictors <- names(my_data)[sapply(my_data, is.numeric)]


# Loop through numeric predictors
for (pred in numeric_predictors) {
  cat("\n\n")
  cat("\n========= ANOVA Test: ", pred, " ~ ", target_var, " =========\n")
  
  # Building formula 
  formula <- as.formula(paste(pred, "~", target_var))
  
  # Performing ANOVA
  anova_model <- aov(formula, data = my_data)
  
  # Displaying summary
  print(summary(anova_model))
}



# ANOVA p-values data frame
anova_results <- data.frame(Predictor = character(), P_Value = numeric())

for (pred in numeric_predictors) {
  formula <- as.formula(paste(pred, "~", target_var))
  model <- aov(formula, data = my_data)
  p_value <- summary(model)[[1]][["Pr(>F)"]][1]
  anova_results <- rbind(anova_results, data.frame(Predictor = pred, P_Value = p_value))
}


# Viewing results
cat(black$bold("============= ANOVA Results: ================\n"))
cat("\n")  
print(anova_results)




##############################################################################




# Combined p-values for chi-square and ANOVA


# Adding method column
chi_results$Method <- "Chi-Square"
anova_results$Method <- "ANOVA"


# Combining as a data frame
combined_results <- rbind(chi_results, anova_results)

# Viewing combined data
cat(black$bold("============= Combined Results: ================\n"))
cat("\n")  
print(combined_results)



custom_colors <- c(
  "ANOVA" = "darkcyan",
  "Chi-Square" = "darkorange"
)


# Bar plot of p-values 

ggplot(combined_results, aes(x = reorder(Predictor, -P_Value), y = P_Value, fill = Method)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  geom_hline(yintercept = 0.05, linetype = "dashed", color = "red") +
  scale_fill_manual(values = custom_colors) + 
  labs(title = "P-Values from Chi-Square and ANOVA Tests",
       x = "Predictor",
       y = "P-Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



# Significant p-values < 0.05
significant_predictors <- combined_results[combined_results$P_Value < 0.05, ]
significant_predictors <- significant_predictors[order(significant_predictors$P_Value), ]


# Viewing significant predictors
cat(black$bold("============= Significant Predictors: ================\n"))
cat("\n")  
print(significant_predictors)



# Basic bar plot for significant predictors

ggplot(significant_predictors, aes(x = reorder(Predictor, P_Value), y = P_Value)) +
  geom_bar(stat = "identity", fill = "darkcyan") +
  coord_flip() +  # Flips axes to make it more readable
  labs(
    title = "Significant Predictors (p < 0.05)",
    x = "Predictor",
    y = "P-Value"
  ) +
  
  theme_minimal()



#############################################################################




# Multinomial Logistic Regression Model with full data


# Target variable to factor 
my_data$Stress.Level <- as.factor(my_data$Stress.Level)


# Setting seed 
set.seed(1234)

# Multinomial Logistic Regression Model 
model_full_data <- multinom(Stress.Level ~ ., data = my_data) 


# Model Summary 
cat("\n\n")
cat(black$bold("============= Model Summary for full data: ================\n"))
cat("\n") 
summary(model_full_data) 



# Predictions with full data

# Removing invalid rows 
my_data <- my_data[my_data$Stress.Level != "", ]
my_data$Stress.Level <- droplevels(factor(my_data$Stress.Level))  # drop the "" level


# Predicting with the cleaned data
predictions <- predict(model_full_data, newdata = my_data)


# Predictions with correct factor levels
predictions <- factor(predictions, levels = levels(my_data$Stress.Level))


cat(black$bold("=============  Model Results with full data: ================\n"))
cat("\n") 


# Confusion matrix 
conf_matrix <- confusionMatrix(predictions, my_data$Stress.Level)
print(conf_matrix)


# Per-class performance 
cat("Class-wise Performance with Full Data:\n") 
print(conf_matrix$byClass) 


# Accuracy metrics 
acc_full_data <- conf_matrix$overall["Accuracy"]
kappa_full_data <- conf_matrix$overall["Kappa"]


# Showing accuracy 
cat("\n\n") 
cat(black$bold("=============  Accuracy Metrics with full data: ================\n"))
cat("\n") 
cat("Accuracy with Full Data: ", round(acc_full_data, 6), "\n") 
cat("Kappa with Full Data: ", round(kappa_full_data, 6), "\n") 



############################################################################




# Train-Test data

# Setting seed 
set.seed(2345)


# Splitting into training (70%) and testing (30%)
train_index <- createDataPartition(my_data$Stress.Level, p = 0.7, list = FALSE)
train_data <- my_data[train_index, ]
test_data <- my_data[-train_index, ]


# Training Multinomial Model 
model_train <- multinom(Stress.Level ~ ., data = train_data) 


# Model Summary 
cat("\n\n")
cat(black$bold("=============  Model Summary with train-test data: ================\n"))
cat("\n") 
summary(model_train)



# Predictions with test data

predictions <- predict(model_train, newdata = test_data) 
cat(black$bold("=============  Model Results with Train and Test Data: ================\n"))
cat("\n") 


# Confusion matrix 
conf_matrix <- confusionMatrix(predictions, test_data$Stress.Level) 
print(conf_matrix) 



# Per-class performance 
cat("Class-wise Performance with Train-Test Data:\n") 
print(conf_matrix$byClass) 


# Accuracy metrics 
acc_train_test <- conf_matrix$overall["Accuracy"]
kappa_train_test <- conf_matrix$overall["Kappa"]


# Showing accuracy 
cat("\n\n") 
cat(black$bold("=============  Accuracy Metrics with Train and Test Data: ================\n"))
cat("\n") 
cat("Accuracy with Tran-Test Data: ", round(acc_train_test, 6), "\n") 
cat("Kappa with Train-Test Data: ", round(kappa_train_test, 6), "\n") 




############################################################################




# Forward Selection 


# Setting seed 
set.seed(3456)

# Null model (intercept only)
null_model <- multinom(Stress.Level ~ 1, data = train_data, trace = FALSE)

# Full model (all predictors)
full_model <- multinom(Stress.Level ~ ., data = train_data, trace = FALSE)

# Stepwise forward selection
forward_model <- stepAIC(null_model, 
                         scope = list(lower = ~1, upper = formula(full_model)),
                         direction = "forward",
                         trace = FALSE)

# Model Summary 
cat(black$bold("=============  Forward Model Summary: ================\n"))
cat("\n") 
summary(forward_model)



# Predicting on test data
forward_predictions <- predict(forward_model, newdata = test_data)
cat(black$bold("=============  Forward Model Results: ================\n"))
cat("\n") 


# Confusion matrix 
forward_conf_matrix <- confusionMatrix(forward_predictions, test_data$Stress.Level)
print(forward_conf_matrix)


# Per-class performance 
cat("Class-wise Performance (Forward Selection):\n")
print(forward_conf_matrix$byClass)


# Accuracy metrics 
acc_forward <- forward_conf_matrix$overall["Accuracy"]
kappa_forward <- forward_conf_matrix$overall["Kappa"]



# Showing Accuracy metrics 
cat("\n\n")
cat(black$bold("=============  Forward Selection: Accuracy Metrics: ================\n"))
cat("\n") 
cat("Accuracy for Forward Selection: ", round(acc_forward, 6), "\n")
cat("Kappa for Forward Selection: ", round(kappa_forward, 6), "\n")




#############################################################################




# Backward Elimination


# Setting seed 
set.seed(3456)


# Starting with full model
backward_model <- stepAIC(full_model,
                          direction = "backward",
                          trace = FALSE)

# Model Summary 
cat(black$bold("=============  Backward Model Summary ================\n"))
cat("\n") 
summary(backward_model)



# Predicting on test data
backward_predictions <- predict(backward_model, newdata = test_data)

cat(black$bold("============= Backward Model Results ================\n"))
cat("\n") 

# Confusion matrix 
backward_conf_matrix <- confusionMatrix(backward_predictions, test_data$Stress.Level)
print(backward_conf_matrix)


# Per-class performance 
cat("Class-wise Performance (Backward Elimination):\n")
print(backward_conf_matrix$byClass)


# Accuracy metrics 
acc_backward <- backward_conf_matrix$overall["Accuracy"]
kappa_backward <- backward_conf_matrix$overall["Kappa"]


# Showing Accuracy metrics 
cat("\n\n")
cat(black$bold("============= Backward Elimination: Accuracy Metrics ================\n"))
cat("\n") 
cat("Accuracy for Backward Elimination: ", round(acc_backward, 6), "\n")
cat("Kappa for Backward Elimination: ", round(kappa_backward, 6), "\n")




##############################################################################




# K-Fold Cross Validation


# Setting seed 
set.seed(4567)


# Defining 5-fold cross-validation
cv_control <- trainControl (
  method = "cv",             # k-fold cross-validation
  number = 5,                # number of folds
  classProbs = TRUE,         # for class probabilities (optional)
  savePredictions = "final", # save all predictions
  summaryFunction = multiClassSummary # performance metrics
)


# Cross-validation Model
multinom_cv_model <- train(
  Stress.Level ~ .,
  data = my_data,
  method = "multinom",
  trControl = cv_control,
  trace = FALSE
)


# Cross-validation model summary
cat(black$bold("============= Cross-validation Model Summary ================\n"))
cat("\n") 
print(multinom_cv_model)



cat(black$bold("============= Cross Validation Model Results ================\n"))
cat("\n") 

# Confusion matrix 
conf_matrix_cv <- confusionMatrix(
  multinom_cv_model$pred$pred,
  multinom_cv_model$pred$obs
)
print(conf_matrix_cv)


# Per-class performance 
cat("Class-wise Performance (K-Fold Cross-Validation):\n")
print(conf_matrix_cv$byClass)


# Accuracy metrics 
acc_kf_cv <- conf_matrix_cv$overall["Accuracy"]
kappa_kf_cv <- conf_matrix_cv$overall["Kappa"]


# Showing Accuracy metrics 
cat("\n\n")
cat(black$bold("============= K-Fold Cross-Validated Accuracy and Kappa ================\n"))
cat("\n") 
cat("Accuracy for K-Fold CV:", round(acc_kf_cv, 6), "\n")
cat("Kappa for K-Fold CV:", round(kappa_kf_cv, 6), "\n")



# Metrics Data frame 

model_comparison <- data.frame(
  Method = c("Full Data Multi", "Train Test Data Model", "Forward Selection", "Backward Elimination", "K-Fold CV"),
  ACCURACY = c(acc_full_data[1], acc_train_test[1], acc_forward[1], acc_backward[1], acc_kf_cv[1]),
  KAPPA = c(kappa_full_data[1], kappa_train_test[1], kappa_forward[1], kappa_backward[1], kappa_kf_cv[1])
  
)

# Printing data frame
cat(black$bold("============= Combined Results: ================\n"))
cat("\n") 
print(model_comparison)





##############################################################################



# Visualizing Combined Results


# Melting the data frame into long format
model_comparison_long <- melt(model_comparison, id.vars = "Method", 
                              variable.name = "Metric", value.name = "Value")


# Defining colors 
custom_colors <- c(
  "ACCURACY" = "darkcyan",
  "KAPPA" = "darkorange"
)

cat(black$bold("============= Combined Results Plot: ================\n"))
cat("\n") 



# Grouped bar plot 

ggplot(model_comparison_long, aes(x = Method, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9), width = 0.9) +
  geom_text(aes(label = sprintf("%.4f", Value)), 
            position = position_dodge(width = 0.9), 
            vjust = -0.3, size = 3) +
  scale_fill_manual(values = custom_colors) +  # Apply custom colors
  labs(title = "Combined Results",
       x = "Modeling Method",
       y = "Metric Value") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )




#############################################################################




# ROC Curve for Cross Validation


# Actual labels to factors 
actual <- as.factor(multinom_cv_model$pred$obs)


# Class probabilities 
probs <- multinom_cv_model$pred[, levels(actual)]


# Initializing empty list 
roc_list <- list()


# ROC for each class
for (class in levels(actual)) {
  roc_list[[class]] <- roc(
    response = as.numeric(actual == class),  # 1 if current class, else 0
    predictor = probs[[class]],
    levels = c(0, 1),
    direction = "<"
  )
}



# Plotting ROC curves

plot(roc_list[[1]], col = 1, main = "Multiclass CV ROC Curves")
for (i in 2:length(roc_list)) {
  plot(roc_list[[i]], col = i, add = TRUE)
}

legend("bottomright", legend = names(roc_list), col = 1:length(roc_list), lwd = 2)




############################################################################

















