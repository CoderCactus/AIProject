#perceptron.py

import numpy as np  # Import NumPy for numerical operations

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        """
        Initializes the perceptron model with:
        - input_size: number of features
        - learning_rate: how much weights are updated
        - epochs: how many times to iterate over the dataset
        """
        self.weights = np.zeros(input_size)  # Initialize weights to 0
        self.bias = 0.0                      # Initialize bias to 0
        self.lr = learning_rate              # Store learning rate
        self.epochs = epochs                 # Store number of training epochs

    def activation(self, x):
        """
        Activation function (step function)
        Returns 1 if input is >= 0, else 0
        """
        return 1 if x >= 0 else 0

    def predict(self, x):
        """
        Predicts the output (0 or 1) for a single input vector `x`
        """
        linear_output = np.dot(self.weights, x) + self.bias  # Weighted sum
        return self.activation(linear_output)                # Apply step function

    def train(self, X_train, y_train):
        """
        Trains the perceptron using the Perceptron learning rule
        - X_train: matrix of input samples
        - y_train: expected outputs
        """
        for epoch in range(self.epochs):  # Repeat for each epoch
            for x, y in zip(X_train, y_train):  # Loop through each training sample
                y_pred = self.predict(x)        # Predict output
                error = y - y_pred              # Calculate the error
                # Update weights and bias using Perceptron rule
                self.weights += self.lr * error * x
                self.bias += self.lr * error

