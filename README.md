# smart water monitoring systems

## Folder Contents
- [water_env](#python environment for the project)
- [Data_Preprocessing.py](#this code is used for data cleaning and dat engineering using tain.csv data)
- [ml.py](# Final model for Water_Consumption prediction)
- [requirement.txt](#install this file for required libraries)
- [submission.csv](#Final predicted Water_Consumption)
- [updated_train.csv](#Cleaned data used for prediction)


## Description

The goal of this project is to develop a Machine Learning model that predicts daily water consumption for individual households based on historical usage patterns, household characteristics, weather conditions, and conservation behaviors.

# Data PreProcessing 
1. Handling Missing or Invalid Values
- Rounding the 'Period_Consumption_Index' column: The values in this column are rounded to two decimal places to ensure consistency.
- Handling invalid 'Income_Level' entries: The code replaces any invalid income levels with 'Unknown'. This ensures that the 'Income_Level' column only contains valid categories or a default 'Unknown' label.
- Handling missing values in 'Appliance_Usage': The missing values in this column are filled with the mode (most frequent value) of the column, as missing values can affect downstream analyses.
- Handling negative values in the 'Guests' column: Any negative values are replaced with 0, as negative guests don't make sense in the context of this dataset.
- Handling missing or invalid values in 'Humidity': Non-numeric values in 'Humidity' are removed, as these would not be suitable for any mathematical model.

2. Feature Transformation
- Stripping extra spaces in 'Apartment_Type': Ensures that any unnecessary spaces around the values are removed, which can affect the consistency of the data.
- Filling missing values in 'Apartment_Type' based on 'Residents': If 'Apartment_Type' is missing, it is filled based on the number of 'Residents'. This is done by a custom function that assigns appropriate values based on predefined conditions.
- Filling missing 'Residents' values based on 'Apartment_Type': If 'Residents' has a negative value, it is corrected based on the 'Apartment_Type'. For example, 'Studio' apartments are assumed to have 1 resident, and other apartment types are adjusted accordingly.

3. Date and Time Processing
- Converting 'Timestamp' to a datetime object: The 'Timestamp' column is converted to a proper datetime format and broken into useful features such as 'Hour', 'Day', 'Month', 'Weekday', and 'Year'.
- Creating time-based features: New columns are derived from the timestamp to help identify patterns related to time (e.g., Is_Weekend, Is_Summer, Is_Day). 

4. Imputation of Missing Values
- Imputing missing 'Temperature' values: A machine learning model (RandomForestRegressor) is used to predict missing 'Temperature' values based on other features such as 'Humidity', 'Water_Price', etc.

5. Feature Encoding
- Label Encoding for categorical features: The 'Apartment_Type' and 'Income_Level' columns are encoded into numerical values using Label Encoding, making them suitable for machine learning models.
- Ordinal Encoding for specific categories: Some features like 'Income_Level' and 'Amenities' are encoded using Ordinal Encoding based on predefined orders.
6. Modeling for Missing Values and Predictions
- Model for 'Income_Level' prediction: The model is trained to predict the 'Income_Level' based on the 'Apartment_Type' for the rows where it is unknown. A RandomForestClassifier is used for this purpose.
- Model for predicting 'Temperature': For missing 'Temperature' values, a regression model is trained to predict the temperature using relevant features such as 'Humidity', 'Water_Price', and other time-related features.

7. Final Cleanup
- Removing unnecessary columns: The 'Timestamp_1' column, which was used for feature extraction, is dropped from the dataset as it is no longer needed.

8. Export the Cleaned Data
- Saving the cleaned dataset: After all the transformations and cleaning, the dataset is saved into a new CSV file (updated_train.csv).

# Water Consumption Prediction
1.# **Model Overview**

The model is built using **XGBoost Regressor**, which is used to predict **Water Consumption** based on several features.

## **Steps Involved:**

### 1. **Dataset Preparation:**
   - The dataset is loaded from `updated_train.csv`, which is assumed to be already cleaned.
   - The target variable (`y`) is **Water_Consumption**, and the remaining columns (except `Timestamp`, `Income_Level`, `Apartment_Type`, and `Amenities`) are used as features.

### 2. **Feature and Target Variables:**
   - **Features (X)**: All columns except `Water_Consumption`, `Income_Level`, `Apartment_Type`, and `Amenities`.
   - **Target (y)**: The `Water_Consumption` column is the target variable.

### 3. **Train-Test Split:**
   - The dataset is split into training and testing sets using `train_test_split()`, with 80% of the data for training and 20% for testing.
   - **Important**: The `Timestamp` column is excluded from the features since it's not directly used for prediction but could be relevant for time-based feature engineering.

### 4. **Hyperparameter Tuning with GridSearchCV:**
   - **XGBoost** is a gradient boosting algorithm suitable for regression tasks.
   - Hyperparameter tuning is performed using **GridSearchCV** to find the optimal hyperparameters. The grid of hyperparameters includes:
     - `n_estimators`: Number of boosting rounds (trees).
     - `learning_rate`: Controls the model's learning speed.
     - `max_depth`: Maximum depth of each tree.
     - `min_child_weight`: Minimum sum of instance weights.
     - `subsample`: Fraction of samples used for each tree.
     - `colsample_bytree`: Fraction of features used for each tree.
   - **GridSearchCV** uses cross-validation to search for the best combination of these parameters.

### 5. **Training the Model:**
   - The best model from the grid search (`grid_search.best_estimator_`) is used to train the model on the training data (`X_train`, `y_train`).

### 6. **Prediction and Evaluation:**
   - After training, the model predicts `Water_Consumption` values on the test set (`X_test`).
   - **Evaluation Metrics**:
     - **Mean Absolute Error (MAE)**: The average absolute difference between predicted and actual values.
     - **Root Mean Squared Error (RMSE)**: Measures the square root of the average of squared differences, which penalizes larger errors more than MAE.

### 7. **Future Predictions:**
   - The model predicts the next 6000 values based on the most recent 6000 rows of data. This is useful for forecasting future water consumption.
   - Predictions are stored in a DataFrame with the corresponding **Timestamp** values from the test set.

### 8. **Saving Predictions:**
   - The predictions for the future timestamps are saved into a new DataFrame (`future_predictions`).
   - The DataFrame is then written to a CSV file called `submission.csv`.

---

## **Key Aspects of the Model:**

- **XGBoost** is used for regression, providing a powerful and efficient way to predict water consumption.
- **GridSearchCV** is used to fine-tune hyperparameters to improve model accuracy.
- **Evaluation Metrics** (MAE and RMSE) are used to assess model performance.
- The model is capable of making future predictions based on recent historical data.



## Installation Instruction

1. pip install requirement.txt
2. run Data_Preprocessing.py on train.csv, cleaned data will be created as updated_train.csv
3. run ml.py for predicting water consumption.


Feel free to adjust this template based on your specific project! Once you provide me with the details, I can update it accordingly.
