# Pokemon Type Prediction

## Overview
This project is part of the Statistics Group Project for the Masters in Data Science at Durham University. The main objective is to create a model that predicts the type of Pokemon based on given data. The dataset includes information about 810 Pokemon from all seven generations, with 41 features each.

## Project Members
- Gudi Dheeraj

## Course Information
- Course: Machine Learning (23/24)
- Module Code: MATH42815_2023
- University: Durham University, UK

## Table of Contents
1. [Introduction](#introduction)
2. [Data Cleaning and Exploratory Data Analysis](#data-cleaning-and-exploratory-data-analysis)
3. [Modelling and Validation](#modelling-and-validation)
    - [Model 1: Classification Trees](#model-1-classification-trees)
    - [Model 2: Random Forests](#model-2-random-forests)
4. [Conclusion](#conclusion)
5. [Bibliography](#bibliography)

## Introduction
The goal of this assignment is to use machine learning techniques to classify the type of Pokemon. Each Pokemon has 41 features, which describe their strengths, abilities, powers, and other characteristics. The dataset contains a mix of numerical and categorical data, presenting a challenge for analysis.

## Data Cleaning and Exploratory Data Analysis
The dataset is read from the `pokemon.csv` file. Initial steps include:
- Reordering columns for better readability.
- Identifying and handling missing values by substituting with modes.
- Converting data types where necessary.
- Removing columns with excessive null values or irrelevant information.
- Handling outliers by examining their impact on the dataset.
- Creating new features where appropriate.
- Normalizing the data before applying machine learning models.

## Modelling and Validation

### Model 1: Classification Trees
#### Data Preparation
- Split the data into training (75%) and test sets (25%).

#### Hyperparameter Tuning
- Parameters tuned include `minsplit`, `minbucket`, and `maxdepth`.
- Cross-validation is used to find the best combination of parameters.
- The best model parameters found are `minsplit = 10`, `minbucket = 5`, and `maxdepth = 9`.

#### Model Building
- The final model is built using the optimal parameters.
- The accuracy of the model on the test data is 80.6%.

#### Model Evaluation
- A confusion matrix is used to evaluate the model performance.

### Model 2: Random Forests
#### Model Training
- A Random Forest model is trained with 500 trees.

#### Model Evaluation
- The model achieves an accuracy of 93.03% on the test data.
- The confusion matrix and other evaluation metrics indicate the superior performance of the Random Forest model over the Classification Tree model.

## Conclusion
Random Forests outperformed Classification Trees in predicting Pokemon types, achieving an accuracy of 93.03% compared to 80.6%. The models were chosen for their ability to handle complex and diverse features in the dataset. Future work could explore combining type1 and type2 for a more comprehensive classification model.

## Bibliography
- Banik, R. (2017). The Complete Pokemon Dataset, Kaggle. Available at: [Kaggle Dataset](https://www.kaggle.com/datasets/rounakbanik/pokemon) (Accessed: 12 February 2024).

---

## Repository Structure
- `data/`: Folder containing the dataset (`pokemon.csv`).
- `scripts/`: R scripts used for data cleaning, EDA, and modeling.
- `models/`: Saved models and results.
- `README.md`: Project overview and instructions.

## Installation and Usage
1. Clone the repository: `git clone https://github.com/GudiDheeraj/Pokemon-classification-using-ML.git`
2. Navigate to the project directory: `cd pokemon-type-prediction`
3. Install the required packages: `pip install -r requirements.txt`
4. Run the scripts in the `scripts/` directory for data cleaning, EDA, and modeling.

---

For more details, refer to the individual sections above.
