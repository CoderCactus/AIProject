import numpy as np

# === Load the dataset ===
data = np.load('letters_training_set_10x.npz')
X = data['inputs']      # Shape: (N, 25)
T = data['targets']     # Shape: (N, 26)
letters_list = data['letters']

class Perceptron:
    def __init__(self, learning_rate):
        self.lr = learning_rate
        self.weights = None

    def train(self):
        # implement training logic
        pass

    def predict(self, x):
        # implement prediction logic
        return 0  # example result


class WidrowHoff:
    def __init__(self, X, T, learning_rate, epochs):
        
        self.input_size = X.shape[1]
        self.output_size = T.shape[1]     # 26 (A–Z)
        self.lr = learning_rate
        self.weights = np.random.uniform(-0.1, 0.1, size=(self.input_size, self.output_size))
        self.epochs = 10000

        X = np.hstack([X, np.ones((X.shape[0], 1))])


    def train(self):
        # implement training logic
        for epoch in range(self.epochs):
            for i, (x, t) in enumerate(zip(X, T)):
                y = np.dot(x, self.weights)
                error = t - y
                delta = self.lr * np.outer(x, error)
                self.weights += delta

        print("Training complete!")

    def predict(self, input_vector):
        # implement prediction logic

        input_with_bias = np.append(input_vector, 1.0)
        output = np.dot(input_with_bias, self.weights)
        return output

# === Test prediction ===
def test_example(index=0):
    model = WidrowHoff(X, T, 0.05, 100)

    model.train()

    test_input = X[index][:-1]  # Remove bias for test
    target_letter = letters_list[np.argmax(T[index])]
    output = model.predict(test_input)
    predicted_index = np.argmax(output)
    predicted_letter = letters_list[predicted_index]

    print(f"Actual Letter:    {target_letter}")
    print(f"Predicted Letter: {predicted_letter}")
    print(f"Output Vector:    {np.round(output, 3)}")

# Run a test
test_example(10)
