# train.py

import pandas as pd               # For reading Excel files
import numpy as np                # For numerical arrays
from perceptron import Perceptron # Import the Perceptron class

# Load the Excel data file (adjust the path if needed)
df = pd.read_excel('data/Perceptron 2022.xls')

# Print the first few rows to understand the structure
print("Data Preview:")
print(df.head())

# Extract input features and labels
# Assuming: all columns except the last are features, last column is the label
X = df.iloc[:, :-1].values.astype(float)  # All rows, all columns except last
y = df.iloc[:, -1].values.astype(int)     # All rows, last column (target)

# Create a Perceptron instance
# input_size is the number of features in each training sample
model = Perceptron(input_size=X.shape[1], learning_rate=0.1, epochs=100)

# Train the perceptron on the data
model.train(X, y)

# Test the trained model on the first sample
sample = X[0]
prediction = model.predict(sample)
print(f"\nTest sample prediction:")
print(f"Predicted: {prediction} | Actual: {y[0]}")

