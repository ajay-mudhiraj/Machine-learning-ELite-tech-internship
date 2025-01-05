# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 22:53:44 2025

@author: Ajay
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data (you can replace this with a dataset of your own)
data = {
    'SquareFootage': [1500, 1800, 2400, 3000, 3500, 4000, 2200, 2600],
    'Bedrooms': [3, 4, 3, 5, 4, 5, 3, 4],
    'Bathrooms': [2, 3, 2, 4, 3, 4, 2, 3],
    'Price': [250000, 320000, 400000, 500000, 550000, 600000, 350000, 450000]
}

# Convert data to a DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Predict the price for a new house
new_house = pd.DataFrame({
    'SquareFootage': [2700],
    'Bedrooms': [4],
    'Bathrooms': [3]
})

predicted_price = model.predict(new_house)
print("Predicted Price for the new house:", predicted_price[0])
