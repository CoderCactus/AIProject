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

class WidrowHoff:
    def __init__(self, X, T, learning_rate, epochs, variable):
        self.lr = learning_rate
        self.epochs = epochs
        self.X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add bias
        self.T = T
        self.input_size = self.X.shape[1]
        self.output_size = T.shape[1]
        self.weights = np.random.uniform(-0.01, 0.01, size=(self.input_size, self.output_size))
        self.variable = variable
        print(f"Initialized Widrow-Hoff model with learning rate {self.lr}, epochs {self.epochs}")
        print(f"Input size: {self.input_size}, Output size: {self.output_size}")

    def train(self):
        print("\nStarting training...\n")
        for epoch in range(1, self.epochs + 1):
            for x, t in zip(self.X, self.T):
                y = np.dot(x, self.weights)
                error = t - y
                self.weights += self.lr * np.outer(x, error)

            if epoch % (self.epochs // 10) == 0 or epoch == 1:

                if self.variable:
                    self.lr *= 0.95

                st.info(f"Epoch {epoch}/{self.epochs} complete")

        st.info("\nTraining complete!")

    def save(self, filename="model.npz"):
        np.savez(filename,
                 weights=self.weights,
                 learning_rate=self.lr,
                 epochs=self.epochs,
                 input_size=self.input_size,
                 output_size=self.output_size)
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename, X, T, variable):
        data = np.load(filename)
        model = cls(X, T, data['learning_rate'], int(data['epochs'], data['variable']))
        model.weights = data['weights']
        print(f"Model loaded from {filename}")
        return model

    def predict(self, input_vector):
        input_with_bias = np.append(input_vector, 1.0)
        return np.dot(input_with_bias, self.weights)


# Test prediction 
def test_example(index=0):
    print(f"\n=== Running Test Example: Index {index} ===")
    model = WidrowHoff(X, T, learning_rate=0.005, epochs=20000)
    model.train()

    test_input = X[index]
    target_letter = letters_list[np.argmax(T[index])]
    output = model.predict(test_input)
    predicted_index = np.argmax(output)
    predicted_letter = letters_list[predicted_index]

    print("\n--- Classification Result ---")
    print(f"Actual Letter:    {target_letter}")
    print(f"Predicted Letter: {predicted_letter}")
    print(f"Output Vector:    {np.round(output, 3)}")
    

    evaluate_model(model, X, T, letters_list)


def evaluate_model(model, X, T, letters_list):
    print("\nEvaluating model on entire dataset...")
    correct = 0
    total = len(X)
    for i in range(total):
        output = model.predict(X[i])
        predicted_index = np.argmax(output)
        actual_index = np.argmax(T[i])
        if predicted_index == actual_index:
            correct += 1
        if i % 100 == 0 and i != 0:
            print(f"Evaluated {i} samples...")

    accuracy = correct / total
    print(f"\n✅ Test Accuracy: {accuracy:.2%}")


# Run a test
#test_example(8)
#
