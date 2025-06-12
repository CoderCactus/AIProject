import numpy as np #Import NumPy for numerical operations
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




# Define the Perceptron class
class Perceptron:
    def __init__(self, X, T, learning_rate=0.01, epochs=1000):
        """
        Initializes the perceptron model with:
        - learning_rate: how much to update weights during training
        - epochs: how many times to iterate over the training data
        """
        self.X = X.flatten()
        self.T = T.flatten()
        n_features = X.shape[1]
        self.bias = 0
        self.weights = np.random.uniform(-0.01, 0.01, size=n_features)
        self.lr = learning_rate                # Store learning rate
        self.epochs = epochs                 # Store number of iterations

        assert X.shape[1] == self.weights.shape[0], "Mismatch in input feature size"


    def activation_func(self, x):
        return np.where(x > 0, 1, 0)

    # Fit the model to the training data
    def fit(self):

        # Convert all y values to 0 or 1 (in case they are -1 or other values)
        T_ = np.where(self.T > 0, 1, 0)

        # Training loop
        for epoch in range(self.epochs):
            for x, t in zip(self.X, self.T):
                # Calculate the linear output: dot product of weights and inputs + bias
                linear_output = np.dot(x, self.weights) + self.bias
                # Apply the activation function (step function)
                T_predicted = self.activation_func(linear_output)

                # Update rule: adjust weights and bias if prediction is wrong
                update = (t - T_predicted)

                self.weights += self.lr * update * x

                self.bias += update

            if epoch % (self.epochs // 10) == 0 or epoch == 1:


                st.info(f"Epoch {epoch}/{self.epochs} complete")

    # Predict the output class for new input data
    def predict(self, x):
        """
        Predicts the binary output for input x.
        """
        linear_output = np.dot(self.weights, x) + self.bias

        return self.activation_func(linear_output)


class MultiClassPerceptron:
    def __init__(self, n_classes, learning_rate=0.01, epochs = 1000):
        self.n_classes = n_classes
        self.models = [
            Perceptron(learning_rate = learning_rate, epochs = epochs)
            for _ in range(n_classes)
        ]

    def fit(self, X, T_onehot):
        for i in range(self.n_classes):
            st.info(f"Training class {i}")
            binary_targets = T_onehot[:, i]
            self.models[i].fit(X, binary_targets)
            

    def predict(self, X):
        # Get raw activation scores from each Perceptron
        scores = np.array([
            np.dot(X, model.weights) + model.bias
            for model in self.models
        ]).T  # Shape: (n_samples, n_classes)

        return scores  # Just the raw output vector



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
                 output_size=self.output_size,
                 variable=self.variable)
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename, X, T, variable):
        data = np.load(filename)
        model = cls(X, T, data['learning_rate'], int(data['epochs']), data['variable'])
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
def test_multiperceptron(n):
    n_classes = T.shape[1]
    model = MultiClassPerceptron(n_classes, learning_rate=0.01, epochs=500)
    model.fit(X, T)

    # Predict a single example
    pred = model.predict(X[n])  # returns one-hot
    predicted_letter = letters_list[np.argmax(pred)]
    actual_letter = letters_list[np.argmax(T[n])]

    print(f"Actual: {actual_letter}, Predicted: {predicted_letter}")

#test_multiperceptron()

def test_perceptron(n):
    n_classes = T.shape[1]
    model = Perceptron(X, T, learning_rate=0.01, epochs=500)
    model.fit()

    # Predict a single example
    pred = model.predict(X[n])  # returns one-hot
    predicted_letter = letters_list[np.argmax(pred)]
    actual_letter = letters_list[np.argmax(T[n])]

    print(f"Actual: {actual_letter}, Predicted: {predicted_letter}")

#test_perceptron(0)