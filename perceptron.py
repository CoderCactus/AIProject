import numpy as np
import streamlit as st
import pandas as pd
import os

# Load the dataset
print("Loading dataset...")
data = np.load('TrainingSets/synthetic_letter_dataset_20x20_50each.npz')
X = data['inputs'] / 255.0    # Normalize pixel values
T = data['targets']
letters_list = data['letters']
print(f"Dataset loaded. Total samples: {X.shape[0]}, Input size: {X.shape[1]}, Number of classes: {T.shape[1]}\n")

# perceptron.py

import numpy as np  # Import NumPy for numerical operations

# Define the activation function: returns 1 if input > 0, else returns 0
def unit_step_func(x):
    return np.where(x > 0, 1, 0)

# Define the Perceptron class
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        """
        Initializes the perceptron model with:
        - learning_rate: how much to update weights during training
        - n_iters: how many times to iterate over the training data
        """
        self.lr = learning_rate                # Store learning rate
        self.n_iters = n_iters                 # Store number of iterations
        self.weights = 0                    # Will be initialized during training
        self.bias = 0                     # Will also be initialized during training
        self.activation_func = unit_step_func  # Use the unit step function for activation

    # Fit the model to the training data
    def fit(self, X, T):
        """
        Trains the perceptron using the Perceptron learning rule.
        Parameters:
        - X: input features, shape (n_samples, n_features)
        - y: target labels (binary, e.g. 0 or 1)
        """
        n_samples=X.shape[0]
        n_features = X.shape[1]
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Convert all y values to 0 or 1 (in case they are -1 or other values)
        T_ = np.where(T > 0, 1, 0)

        # Training loop
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Calculate the linear output: dot product of weights and inputs + bias
                linear_output = np.dot(x_i, self.weights) + self.bias
                # Apply the activation function (step function)
                T_predicted = self.activation_func(linear_output)

                # Update rule: adjust weights and bias if prediction is wrong
                update = self.lr * (T_[idx] - T_predicted)
                self.weights += update * x_i
                self.bias += update

    # Predict the output class for new input data
    def predict(self, x):
        """
        Predicts the binary output for input `x`.
        """
        linear_output = np.dot(self.weights, x) + self.bias



