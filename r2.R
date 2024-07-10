library(DescTools)
library(rpart)
library(caret)
library(e1071)
library(dplyr)
library(ggplot2)
library(corrplot)
library(rpart.plot)
library(rpart)
library(caret)
library(randomForest)
library(partykit)
library(randomForestExplainer)
library(dplyr)
library(cowplot)

#read data
data <- read.csv("C:/Users/dheer/OneDrive/Desktop/pokemon.csv")
data
dim(data)
summary(data)
column_index <- which(names(data) == "name")
#unique classifications
unique_names <- unique(data$classfication)
unique_names
# Print unique names
print(unique_names)
# Reorder the columns
data <- data[, c(column_index, setdiff(1:ncol(data), column_index))]

# Check for missing values column-wise
null_values_by_column <- colSums(is.na(data))
null_values_by_column
# Print column names and number of null values
for (col in names(null_values_by_column)) {
  if (null_values_by_column[col] > 0) {
    cat("Column:", col, "- Number of null values:", null_values_by_column[col], "\n")
  }
}

# Function to replace NA values with the mode of the column
replace_with_mode <- function(x) {
  mode_value <- as.numeric(names(sort(table(x), decreasing = TRUE)[1]))
  x[is.na(x)] <- mode_value
  return(x)
}

# Replace percentage_male values with NULL


# Replace weight_kg with the mode of the column
data$height_m <- replace_with_mode(data$height_m)
data$weight_kg <- replace_with_mode(data$weight_kg)
data$percentage_male <- replace_with_mode(data$percentage_male)
# Replace empty values with NA in the 'type2' column

# Print the updated data frame
View(data)
str(data)

#we dont need japanese name and pokedex number
print(data$capture_rate)
data$capture_rate[774] <- 30
data$capture_rate <- as.integer(data$capture_rate)#
data <- subset(data, select = -japanese_name)
data <- subset(data, select = -pokedex_number)
data <- subset(data, select = -type2)
data$top_abilities <- sapply(data$abilities, function(x) length(strsplit(x, " ")[[1]]))

# Print or view the modified data frame
print(data)


dim(data)
columns_with_na <- colSums(is.na(data)) > 0

# Display columns with NA values
names(data)[columns_with_na]







# Create a new column 'top_abilities' containing the count of words in each element of the 'abilities' column
data$top_abilities <- sapply(data$abilities, function(x) length(strsplit(x, " ")[[1]]))

# Print or view the modified data frame
print(data)



# Function to detect outliers using IQR method
detect_outliers <- function(x) {
  Q1 <- quantile(x, 0.25)
  Q3 <- quantile(x, 0.75)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  outliers <- x[x < lower_bound | x > upper_bound]
  return(length(outliers))
}

# Initialize vectors to store column names and outlier counts
column_names <- c()
outlier_counts <- c()

# Loop through each column of the data frame and detect outliers
for (col in names(data)) {
  if (is.numeric(data[[col]])) {
    outlier_count <- detect_outliers(data[[col]])
    column_names <- c(column_names, col)
    outlier_counts <- c(outlier_counts, outlier_count)
  } else {
    cat("Column", col, "is not numeric, skipping outlier detection.\n\n")
  }
}

# Create a data frame for plotting
outliers_df <- data.frame(Column = column_names, Outliers = outlier_counts)

# Plot the number of outliers in each column
ggplot(outliers_df, aes(x = Column, y = Outliers)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black") +
  labs(title = "Number of Outliers in Each Column", x = "Column", y = "Number of Outliers") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_text(aes(label = Outliers), vjust = -0.3, size = 3)

total_unique_types <- length(unique(data$type1))

# Print the total number of unique types
print(total_unique_types)

selected_data <- data[, c("attack", "defense", "experience_growth", "hp", "speed")]
# Compute the correlation matrix for the selected columns


# Extract unique values of "type1" from the data frame
unique_types <- unique(data$type1)
unique_types
summary(data$speed)
# Modify the ggplot code
data1 <- data.frame(category = data$type1)

# Convert categories to numeric labels
data1$numeric_label <- as.numeric(factor(data1$category))


df <- data
type_1_counts <- table(df$type1)



# Convert the result to a data frame
type_1_counts_df <- data.frame(Type_1 = names(type_1_counts), Count = as.numeric(type_1_counts))

# Plot the bar chart
ggplot(type_1_counts_df, aes(x = reorder(Type_1, -Count), y = Count, fill = Type_1)) +
  geom_bar(stat = "identity") +
  labs(title = "Number of Pokemon by Type 1",
       x = "Type 1",
       y = "No of Pokemon") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = rainbow(18)) +
  coord_flip()



# Print the modified data frame
print(df)
View(df)
exclude_columns <- c("name", "classfication","type1","abilities")

# Exclude specified columns
df_cor <- df[, !(names(df) %in% exclude_columns)]

# Calculate correlation matrix
correlation_matrix <- cor(df_cor)
highly_correlated_pairs <- which(abs(correlation_matrix) > 0.65 & correlation_matrix != 1, arr.ind = TRUE)

# Get the names of variables that are highly correlated
variable_names <- rownames(correlation_matrix)[highly_correlated_pairs[,1]]
correlated_variable_names <- cbind(variable_names, colnames(correlation_matrix)[highly_correlated_pairs[,2]])

# Print the highly correlated variable pairs
print("Highly correlated variables:")
print(correlated_variable_names)
type_1 <- as.factor(data$type1)
str(data)
# Scatter plot against_ghost vs against_dark



color_scale <- scale_color_manual(values = rainbow(18), labels = unique_types)

# Create scatter plots for each combination of variables
scatter_plots <- list(
  ggplot(data, aes(x = against_ghost, y = against_dark, color = factor(numeric_label))) +
    geom_point() +
    labs(title = "Scatter Plot", x = "against_ghost", y = "against_dark ") +
    color_scale,
  
  ggplot(data, aes(x = base_total, y = attack, color = factor(numeric_label))) +
    geom_point() +
    labs(title = "Scatter Plot", x = "base total", y = "attack ") +
    color_scale,
  
  ggplot(data, aes(x = base_total, y = capture_rate, color = factor(numeric_label))) +
    geom_point() +
    labs(title = "Scatter Plot", x = "base_total", y = "capture_rate") +
    color_scale,
  
  ggplot(data, aes(x = base_total, y = sp_attack, color = factor(numeric_label))) +
    geom_point() +
    labs(title = "Scatter Plot", x = "base_total", y = "sp_attack") +
    color_scale,
  
  ggplot(data, aes(x = base_total, y = sp_defense, color = factor(numeric_label))) +
    geom_point() +
    labs(title = "Scatter Plot", x = "base_total", y = "sp_defense") +
    color_scale
)

# Combine scatter plots into a single figure
combined_plot <- cowplot::plot_grid(plotlist = scatter_plots, nrow = 3)

# Print the combined plot
print(combined_plot)



corrplot(correlation_matrix, method = "shade", type = "upper", tl.pos = "lt",linecolor = "black")# Plot correlation heatmap
heatmap(correlation_matrix,
        col = colorRampPalette(c <- rainbow(18))(100),  # Choose a color palette
        symm = TRUE,  # Show symmetrical range around zero
        margins = c(10, 10))  # Add extra space for labels
#bast total is totally correlated to sp_attack and sp_defence
#base total and attack are corelated
#base egg steps and is legendery are corelated
#height_m and weight_kg are correlated
df <- data %>%
  mutate(sum_column = rowSums(select(., 3:20)))

columns_to_remove <- c("sp_attack", "sp_defence","base_egg_steps","height_m","name", "classfication","abilities","against_ghost","attack","capture_rate")
df <- df[, !(names(df) %in% columns_to_remove)]
View(df)
type_column <- df$type1

# Remove the 'type' column before normalization
df_without_type <- df[, -which(names(df) == "type1")]
View(df_without_type)
# Normalize the dataframe (excluding the 'type' column)
normalized_data <- scale(df_without_type)

# Combine the 'type' column with the normalized data
normalized_df <- cbind(type1 = type_column, as.data.frame(normalized_data))

# Output the normalized dataframe
View(normalized_df)

df<-normalized_df




set.seed(123)  # Set seed for reproducibility
#split the train and test data
train_index <- sample(1:nrow(df), 0.75 * nrow(df))
train_data <- df[train_index, ]
test_data <- df[-train_index, ]




dev.off()  # Close all open graphics devices

# Build the classification tree model







# Define the grid of parameters
param_grid <- expand.grid(
  minsplit = c(10, 20, 30),
  minbucket = c(5, 10, 15),
  maxdepth = c(3,6,9,12,15)
)

# Initialize variables to store the best model and its performance
best_accuracy <- 0
best_model <- NULL

# Initialize vectors to store accuracies
accuracies <- c()

# Loop through each combination of parameters
for (i in 1:nrow(param_grid)) {
  minsplit_val <- param_grid$minsplit[i]
  minbucket_val <- param_grid$minbucket[i]
  maxdepth_val <- param_grid$maxdepth[i]
  
  # Train the model with current parameter values
  tree_model <- rpart(
    type1 ~ .,
    data = train_data,
    method = "class",
    minsplit = minsplit_val,
    minbucket = minbucket_val,
    control = rpart.control(maxdepth = maxdepth_val)
  )
  
  # Perform cross-validation
  set.seed(123) # For reproducibility
  folds <- createFolds(train_data$type1, k = 5, list = TRUE)
  accuracy <- 0
  
  for (j in 1:5) {
    train_indices <- unlist(folds[-j])
    test_indices <- unlist(folds[j])
    train_fold <- train_data[train_indices, ]
    test_fold <- train_data[test_indices, ]
    
    # Train the model on the training fold
    fold_model <- rpart(
      type1 ~ .,
      data = train_fold,
      method = "class",
      minsplit = minsplit_val,
      minbucket = minbucket_val,
      control = rpart.control(maxdepth = maxdepth_val)
    )
    
    # Make predictions on the test fold
    predictions <- predict(fold_model, test_fold, type = "class")
    
    # Calculate accuracy
    accuracy <- accuracy + sum(predictions == test_fold$type1) / length(predictions)
  }
  
  # Average accuracy across folds
  accuracy <- accuracy / 5
  
  # Store accuracy in the vector
  accuracies <- c(accuracies, accuracy)
}

# Plot accuracies for different parameter combinations
plot(1:nrow(param_grid), accuracies, type = "b", xlab = "Parameter Combination", ylab = "Accuracy", main = "Accuracy for Different Parameter Combinations")

# Find the index of the best accuracy
best_index <- which.max(accuracies)

# Add a point to indicate the best accuracy
points(best_index, accuracies[best_index], col = "red", pch = 19)

# Add text to label the best accuracy point
text(best_index, accuracies[best_index], labels = paste("Best Accuracy:", round(accuracies[best_index], 4)), pos = 3, col = "red")


# Print best parameters and accuracy
print(paste("Best Parameters:", paste(names(best_params), best_params, sep = "=", collapse = ", ")))

cv_model <- rpart.control(cp = seq(0.01, 0.5, by = 0.01))  # Range of cp values to try
cv_results <- rpart(type1 ~ ., data = train_data, method = "class", control = cv_model)

# Print the cross-validation results
printcp(cv_results)

tree_model <- rpart(type1 ~ .,
                    data = train_data,
                    method = "class",
                    control = rpart.control(cp = 0.01, minsplit = 10, minbucket = 5, maxdepth = 9))
rpart.plot(tree_model, box.palette="RdBu", digits=-1)
plot(as.party(tree_model))
plot(tree_model)
predictions <- predict(tree_model, test_data, type = "class")
conf_matrix <- confusionMatrix(factor(predictions), factor(test_data$type1))
print(conf_matrix)


# Access the frame element of the tree model
tree_frame <- tree_model$frame

# Extract the variable names from the frame
used_variables <- unique(tree_frame$var)
used_variables

# Train a random forest model
rf_model <- randomForest(factor(type1) ~., data = train_data)
plot(rf_model)
rf_model
predictions <- predict(rf_model, newdata = test_data)
factor(predictions)
# Calculate the confusion matrix
conf_matrix <- confusionMatrix(factor(predictions), factor(test_data$type1))

# Print the confusion matrix
print(conf_matrix)

# Optionally, print other performance metrics
print(conf_matrix$overall)

# Obtain the number of trees in the Random Forest model
num_trees <- length(rf_model$forest)



# Initialize variables to track the best tree
best_tree_index <- 1
best_tree_error <- rf_model$err.rate[1, "OOB"]

# Loop through the trees in the Random Forest model to find the one with the lowest out-of-bag error
for (i in 2:num_trees) {
  if (rf_model$err.rate[i, "OOB"] < best_tree_error) {
    best_tree_index <- i
    best_tree_error <- rf_model$err.rate[i, "OOB"]
  }
}

# Extract the best tree
individual_tree <- getTree(rf_model, best_tree_index, labelVar = TRUE)
plot(individual_tree)
