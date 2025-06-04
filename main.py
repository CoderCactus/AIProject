import numpy as np

# === Load the dataset ===
data = np.load('synthetic_letter_dataset_20x20_10each.npz')
X = data['inputs'] / 255.0    # Normalize pixel values
T = data['targets']
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
        self.lr = learning_rate
        self.epochs = epochs
        self.X = np.hstack([X, np.ones((X.shape[0], 1))])
        self.T = T
        self.input_size = self.X.shape[1]
        self.output_size = T.shape[1]
        self.weights = np.random.uniform(-0.01, 0.01, size=(self.input_size, self.output_size))


    def train(self):
        for epoch in range(self.epochs):
            for x, t in zip(self.X, self.T):
                y = np.dot(x, self.weights)
                error = t - y
                self.weights += self.lr * np.outer(x, error)
        print("Training complete!")

    def predict(self, input_vector):
        input_with_bias = np.append(input_vector, 1.0)
        return np.dot(input_with_bias, self.weights)


# === Test prediction ===
def test_example(index=0):
    model = WidrowHoff(X, T, learning_rate=0.001, epochs=50000)
    model.train()

    test_input = X[index]
    target_letter = letters_list[np.argmax(T[index])]
    output = model.predict(test_input)
    predicted_index = np.argmax(output)
    predicted_letter = letters_list[predicted_index]

    print(f"Actual Letter:    {target_letter}")
    print(f"Predicted Letter: {predicted_letter}")
    print(f"Output Vector:    {np.round(output, 3)}")

    
    evaluate_model(model, X, T, letters_list)

def evaluate_model(model, X, T, letters_list):
    correct = 0
    total = len(X)
    for i in range(total):
        output = model.predict(X[i])
        predicted_index = np.argmax(output)
        actual_index = np.argmax(T[i])
        if predicted_index == actual_index:
            correct += 1
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.2%}")


# Run a test
#test_example(8)



