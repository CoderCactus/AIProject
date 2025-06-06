import numpy as np
import streamlit as st
import pandas as pd
import os

# === Load the dataset ===
print("Loading dataset...")
data = np.load('TrainingSets/synthetic_letter_dataset_20x20_50each.npz')
X = data['inputs'] / 255.0    # Normalize pixel values
T = data['targets']
letters_list = data['letters']
print(f"Dataset loaded. Total samples: {X.shape[0]}, Input size: {X.shape[1]}, Number of classes: {T.shape[1]}\n")

class Perceptron:
    def __init__(self, learning_rate):
        self.lr = learning_rate
        self.weights = None

    def train(self):
        print("Training Perceptron... (Not yet implemented)")
        pass

    def predict(self, x):
        return 0  # example result


class WidrowHoff:
    def __init__(self, X, T, learning_rate, epochs):
        self.lr = learning_rate
        self.epochs = epochs
        self.X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add bias
        self.T = T
        self.input_size = self.X.shape[1]
        self.output_size = T.shape[1]
        self.weights = np.random.uniform(-0.01, 0.01, size=(self.input_size, self.output_size))
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
                st.info(f"Epoch {epoch}/{self.epochs} complete")

        st.info("\nTraining complete!")

    def save(self, filename="model.npz"):
        np.savez(filename,
                 weights=self.weights,
                 learning_rate=self.lr,
                 epochs=self.epochs,
                 input_size=self.input_size,
                 output_size=self.output_size)
        print(f"✅ Model saved to {filename}")

    @classmethod
    def load(cls, filename, X, T):
        data = np.load(filename)
        model = cls(X, T, data['learning_rate'], int(data['epochs']))
        model.weights = data['weights']
        print(f"✅ Model loaded from {filename}")
        return model

    def predict(self, input_vector):
        input_with_bias = np.append(input_vector, 1.0)
        return np.dot(input_with_bias, self.weights)


# === Test prediction ===
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

