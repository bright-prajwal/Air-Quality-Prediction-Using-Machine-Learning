# Load libraries
library(readr)
library(dplyr)
library(ggplot2)
library(caret)
library(rpart)
library(randomForest)

# Load data
data <- read.csv("data.csv", stringsAsFactors = FALSE)

# Filter for Delhi
data <- data %>% filter(state == "Delhi")

# Select relevant columns
data_clean <- data %>%
  select(pm2_5, so2, no2, rspm, spm) %>%
  mutate(across(everything(), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))

cat("Rows after cleaning: ", nrow(data_clean), "\n")


# Split data

set.seed(42)
trainIndex <- createDataPartition(data_clean$pm2_5, p = 0.8, list = FALSE)
train <- data_clean[trainIndex, ]
test <- data_clean[-trainIndex, ]


# UNDERFITTING: Linear Model (use single feature, e.g., so2)

lm_model <- lm(pm2_5 ~ so2, data = train)
data_clean$lm_pred <- predict(lm_model, data_clean)
lm_mse <- mean((test$pm2_5 - predict(lm_model, test))^2)
cat("Linear Model MSE (Underfitting):", lm_mse, "\n")

p_under <- ggplot(data_clean, aes(so2, pm2_5)) +
  geom_point(color = "gray", alpha = 0.6) +
  geom_line(aes(y = lm_pred), color = "blue", size = 1.2) +
  labs(title = "Underfitting Example (Linear Model)", y = "Predicted pm2.5", x = "SO2") +
  theme_minimal(base_size = 14)

print(p_under)
ggsave("Delhi_underfit_linear.png", plot = p_under, width = 8, height = 5, dpi = 150)


# BALANCED FIT: Random Forest

set.seed(42)
rf_model <- randomForest(pm2_5 ~ so2 + no2 + rspm + spm, data = train, ntree = 200)
data_clean$rf_pred <- predict(rf_model, data_clean)
rf_mse <- mean((test$pm2_5 - predict(rf_model, test))^2)
cat("Random Forest MSE:", rf_mse, "\n")

p_rf <- ggplot(data_clean, aes(so2, pm2_5)) +
  geom_point(color = "gray", alpha = 0.6) +
  geom_line(aes(y = rf_pred), color = "green", size = 1.2) +
  labs(title = "Balanced Fit Example (Random Forest)", y = "Predicted pm2.5", x = "SO2") +
  theme_minimal(base_size = 14)

print(p_rf)
ggsave("Delhi_balanced_rf.png", plot = p_rf, width = 8, height = 5, dpi = 150)


# OVERFITTING: Decision Tree (high complexity)

tree_model <- rpart(pm2_5 ~ so2 + no2 + rspm + spm, data = train, control = rpart.control(cp = 0.0001))
data_clean$tree_pred <- predict(tree_model, data_clean)
tree_mse <- mean((test$pm2_5 - predict(tree_model, test))^2)
cat("Decision Tree MSE (Overfitting):", tree_mse, "\n")

p_over <- ggplot(data_clean, aes(so2, pm2_5)) +
  geom_point(color = "gray", alpha = 0.6) +
  geom_line(aes(y = tree_pred), color = "red", size = 1.2) +
  labs(title = "Overfitting Example (Decision Tree)", y = "Predicted pm2.5", x = "SO2") +
  theme_minimal(base_size = 14)

print(p_over)
ggsave("Delhi_overfit_tree.png", plot = p_over, width = 8, height = 5, dpi = 150)

