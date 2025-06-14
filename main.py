import numpy as np #Import NumPy for numerical operations
import streamlit as st
import string


# Load the dataset
print("Loading dataset...")
data = np.load('TrainingSets/synthetic_letter_dataset_20x20_50each.npz')
X = data['inputs'] / 255.0    # Normalize pixel values
T = data['targets']
letters_list = data['letters']
print(f"Dataset loaded. Total samples: {X.shape[0]}, Input size: {X.shape[1]}, Number of classes: {T.shape[1]}\n")




# Define the Perceptron class
class BinaryPerceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=20):
        self.weights = np.random.uniform(-0.01, 0.01, size=input_size)
        self.bias = 0.0
        self.lr = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else -1

    def train(self, X, T):
        # Convert targets from 0/1 to -1/+1 for perceptron
        T = np.where(T > 0, 1, -1).astype(np.float32)

        ib = st.empty()
        for epoch in range(self.epochs):
            for (x, t) in zip(X, T):
                z = np.dot(self.weights, x) + self.bias
                y = 1 if z >= 0 else -1  # Activation for perceptron
                error = t - y
                self.weights += self.lr * error * x
                self.bias += self.lr * error

            if epoch % (self.epochs // 10) == 0 or epoch == 1:
                ib.info(f"Epoch {epoch}/{self.epochs} complete")
        ib.empty()


    def raw_output(self, x):
        return np.dot(self.weights, x) + self.bias

    def predict(self, x):
        return self.activation(self.raw_output(x))

class MultiClassPerceptron:
    def __init__(self, n_classes, input_size, learning_rate=0.001, epochs=100):
        self.n_classes = n_classes
        self.models = [
            BinaryPerceptron(input_size, learning_rate, epochs)
            for _ in range(n_classes)
        ]

    def train(self, X, T_onehot):
        ib = st.empty()
        for class_index in range(self.n_classes):
            ib.info(f"Training perceptron for class {string.ascii_uppercase[class_index]}")
            binary_labels = T_onehot[:, class_index]
            self.models[class_index].train(X, binary_labels)

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # Calculate scores for all samples across all models
        scores = np.array([[model.raw_output(x) for model in self.models] for x in X])  # shape: (n_samples, n_classes)
        return np.argmax(scores, axis=1)


    def save(self, filename="model.npz"):
        weights = np.array([model.weights for model in self.models])
        biases = np.array([model.bias for model in self.models])
        np.savez(filename, weights=weights, biases=biases)

    @classmethod
    def load(cls, filename, input_size, learning_rate=0.001, epochs=100):
        data = np.load(filename)
        weights = data['weights']
        biases = data['biases']
        n_classes = weights.shape[0]
        model = cls(n_classes, input_size, learning_rate, epochs)
        for i in range(n_classes):
            model.models[i].weights = weights[i]
            model.models[i].bias = biases[i]
        return model
    
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
        ib = st.empty()  # create a placeholder outside the loop
        print("\nStarting training...\n")
        for epoch in range(1, self.epochs + 1):
            for x, t in zip(self.X, self.T):
                y = np.dot(x, self.weights)
                error = t - y
                self.weights += self.lr * np.outer(x, error)

            if epoch % (self.epochs // 10) == 0 or epoch == 1:

                if self.variable:
                    self.lr *= 0.95

                ib.info(f"Epoch {epoch}/{self.epochs} complete")

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
def evaluate(model, X, T, letters):
    correct = 0
    total = X.shape[0]  # ✅ Define total based on number of samples

    for i in range(total):
        pred_index = model.predict(X[i])
        if isinstance(pred_index, np.ndarray):
            pred_index = pred_index.item()

        actual_index = np.argmax(T[i])
        pred_letter = letters[pred_index]
        actual_letter = letters[actual_index]

        # If letters are arrays, convert to string
        if isinstance(pred_letter, (np.ndarray, list)):
            pred_letter = pred_letter.item()
        if isinstance(actual_letter, (np.ndarray, list)):
            actual_letter = actual_letter.item()

        if pred_index == actual_index:
            correct += 1

    accuracy = correct / total
    print(f"\n Accuracy: {accuracy:.2%}")

'''
n_classes = T.shape[1]
input_size = X.shape[1]
letters = data['letters']
model = MultiClassPerceptron(n_classes, input_size, learning_rate=0.01, epochs=1000)
indices = np.random.permutation(len(X))
X, T = X[indices], T[indices]
model.train(X, T)

evaluate(model, X, T, letters)
'''