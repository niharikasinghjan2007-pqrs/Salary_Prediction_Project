
# Importing required libraries and checking installed library versions 
import sklearn
print(sklearn.__version__)

import pandas
print(pandas.__version__)

import numpy
print(numpy.__version__)

import matplotlib
print(matplotlib.__version__)

import seaborn
print(seaborn.__version__)


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing machine learning tools from sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score



# Step 1: Loading the Dataset
# Reading the dataset from CSV file
data = pd.read_csv("salary_data.csv")

# Displaying first few rows to understand structure of data
print("Dataset Preview:")
print(data.head())

# Displaying dataset summary including data types and null values
print("\nDataset Info:")
print(data.info())


# Step 2: Data Preprocessing (Encoding Categorical Data)
# Creating separate LabelEncoders for each categorical column
education_encoder = LabelEncoder()
location_encoder = LabelEncoder()
skill_encoder = LabelEncoder()

# Converting categorical text values into numeric format as machine learning models cannot work directly with text data
data["Education_Level"] = education_encoder.fit_transform(data["Education_Level"])
data["Location"] = location_encoder.fit_transform(data["Location"])
data["Skill_Level"] = skill_encoder.fit_transform(data["Skill_Level"])


# Step 3: Splitting Features and Target Variable
# X contains input features (independent variables) and y contains target variable (dependent variable)
X = data.drop("Salary", axis=1)
y = data["Salary"]

print("\nFeatures Used:")
print(X.head())


# Step 4: Train-Test Split
# Splitting data into training (80%) and testing (20%). Also random_state ensures same split every time for consistency
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# Step 5: Random Forest Regression Model
# Initializing Random Forest Regressor
# n_estimators = number of decision trees used in the model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42)

# Training the model on training data
rf_model.fit(X_train, y_train)

# Predicting salary values on test data
rf_predictions = rf_model.predict(X_test)

# Evaluating model performance
print("\nRandom Forest Results :-")
print("Mean Absolute Error:", mean_absolute_error(y_test, rf_predictions))
print("R2 Score:", r2_score(y_test, rf_predictions))


# Step 6: Gradient Boosting Regression Model
# Initializing Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    random_state=42)

# Training the model
gb_model.fit(X_train, y_train)

# Making predictions
gb_predictions = gb_model.predict(X_test)

# Evaluating performance
print("\n----- Gradient Boosting Results -----")
print("Mean Absolute Error:", mean_absolute_error(y_test, gb_predictions))
print("R2 Score:", r2_score(y_test, gb_predictions))


# Step 7: Feature Importance Visualization
# Extracting importance score of each feature from Random Forest
feature_importance = rf_model.feature_importances_

# Plotting feature importance graph
plt.figure()
plt.bar(X.columns, feature_importance)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.show()


# Step 8: Predicting Salary for a New Candidate
# Creating a new data sample for prediction
# Categorical values must be encoded using the same encoders used earlier
new_candidate = pd.DataFrame({
    "Experience": [15],
    "Education_Level": [education_encoder.transform(["Master"])[0]],
    "Location": [location_encoder.transform(["Delhi"])[0]],
    "Skill_Level": [skill_encoder.transform(["Advanced"])[0]]})

# Predicting salary using trained Random Forest model
predicted_salary = rf_model.predict(new_candidate)
print("\nPredicted Salary for New Candidate:-", predicted_salary[0])