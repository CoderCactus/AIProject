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
        self.weights = None                    # Will be initialized during training
        self.bias = None                       # Will also be initialized during training
        self.activation_func = unit_step_func  # Use the unit step function for activation

    # Fit the model to the training data
    def fit(self, X, y):
        """
        Trains the perceptron using the Perceptron learning rule.
        Parameters:
        - X: input features, shape (n_samples, n_features)
        - y: target labels (binary, e.g. 0 or 1)
        """
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Convert all y values to 0 or 1 (in case they are -1 or other values)
        y_ = np.where(y > 0, 1, 0)

        # Training loop
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Calculate the linear output: dot product of weights and inputs + bias
                linear_output = np.dot(x_i, self.weights) + self.bias
                # Apply the activation function (step function)
                y_predicted = self.activation_func(linear_output)

                # Update rule: adjust weights and bias if prediction is wrong
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    # Predict the output class for new input data
    def predict(self, x):
        """
        Predicts the binary output for input `x`.
        """
        linear_output = np.dot(self.weights, x) + self.bias
        return self.activation_func(linear_output)

# If this script is run directly, do a simple test
if __name__ == "__main__":
    # Imports for testing the Perceptron
    import matplotlib.pyplot as plt  # Corrected typo in import
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    # Load a binary classification dataset
    X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Create a perceptron instance and train it
    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(X_train, y_train)

    # Predict on the test set
    predictions = [p.predict(x) for x in X_test]

    # (Optional) Visualization of decision boundary
    # Plotting code can be added here if needed


