import numpy as np # NumPy for numerical operations
import streamlit as st # Streamlit for UI interaction
import string # For alphabetical letter handling


# Load the dataset
print("Loading dataset...")
data = np.load('TrainingSets/synthetic_letter_dataset_20x20_50each.npz') # Load synthetic dataset of letters
X = data['inputs'] / 255.0 # Normalize pixel values to [0, 1]
T = data['targets'] # One-hot encoded labels
letters_list = data['letters'] # Class letter labels (A-Z)
print(f"Dataset loaded. Total samples: {X.shape[0]}, Input size: {X.shape[1]}, Number of classes: {T.shape[1]}\n")


# Defines the Perceptron class (single layer binary)
class BinaryPerceptron:
    """
    Binary perceptron using ±1 as targets and prediction outputs.
    """

    def __init__(self, input_size, learning_rate=0.01, epochs=20):
        """
        Initialize binary perceptron with random weights and specified learning parameters.
        """

        self.weights = np.random.uniform(-0.01, 0.01, size=input_size) # Initialize random weights
        self.bias = 0.0 # Bias term initialized to zero
        self.lr = learning_rate # Define learning rate
        self.epochs = epochs # Define number of training iterations


    def activation(self, x):
        """
        Step activation function for binary classification.
        Returns 1 if input is non-negative, else -1.
        """

        return 1 if x >= 0 else -1

    def train(self, X, T):
        """
        Train the perceptron using the Perceptron learning rule.
        Converts 0/1 labels to -1/1 before training.
        """

        # Convert targets from 0/1 to -1/+1 for perceptron
        T = np.where(T > 0, 1, -1).astype(np.float32)
        ib = st.empty() # Streamlit status placeholder
        for epoch in range(self.epochs):
            for (x, t) in zip(X, T):
                z = np.dot(self.weights, x) + self.bias # Define linear combination
                y = 1 if z >= 0 else -1  # Activation for perceptron
                error = t - y # Compute prediction error
                self.weights += self.lr * error * x # Update weights
                self.bias += self.lr * error # Update bias

             # Periodically update training progress
            if epoch % (self.epochs // 10) == 0 or epoch == 1:
                ib.info(f"Epoch {epoch}/{self.epochs} complete")
        ib.empty() # Clear Streamlit output

    def predict(self, x):
        '''
        Predict the output class for new input data (Input x)
        '''

        linear_output = np.dot(self.weights, x) + self.bias
        return self.activation_func(linear_output)

    def raw_output(self, x):
        """
        Calculate raw output (before activation).
        """

        return np.dot(self.weights, x) + self.bias

    def predict(self, x):
        """
        Return the predicted binary class for input vector x.
        """

        return self.activation(self.raw_output(x))

# Multi-Class Perceptron using One-vs-All
class MultiClassPerceptron:
    """
    Multi-class classification using multiple binary perceptrons in a one-vs-all setup.
    """

    def __init__(self, n_classes, input_size, learning_rate=0.001, epochs=100, variable=False):
        """
        Initialize multiple binary perceptrons for multi-class classification.
        """

        self.n_classes = n_classes # Total number of unique classes
        self.learning_rate = learning_rate # Set the learning rate for all perceptrons
        self.epochs = epochs # Set the number of epochs

        self.models = []  # List to hold individual Perceptron models, one for each class

    def train(self, X, T_onehot):
        """
        Train each binary perceptron to recognize one class versus all others.
        """

        ib = st.empty() # Create a placeholder in Streamlit for progress updates

        # Loop over all classes to train one binary perceptron per class
        for class_index in range(self.n_classes):
            ib.info(f"Training perceptron for class {string.ascii_uppercase[class_index]}")
            binary_labels = T_onehot[:, class_index] # Extract 0/1 labels for this class
            self.models[class_index].train(X, binary_labels) # Train corresponding model

    def predict(self, X):
        """
        Predict class index by choosing the perceptron with the highest raw output.
        """

        if X.ndim == 1:
            X = X.reshape(1, -1) # Ensure input is in batch format for prediction
        scores = np.array([[model.raw_output(x) for model in self.models] for x in X])  # Calculate scores for all samples across all models
        return np.argmax(scores, axis=1) # Return class index with highest score

      
    def save(self, filename="model.npz"):
        """
        Save model parameters (weights and biases) to file.
        """

        # Save weights and biases for all models (one per class)
        weights = np.array([model.weights for model in self.models])
        biases = np.array([model.bias for model in self.models])

        # Store model parameters in a .npz file
        np.savez(filename,
                weights=weights,
                biases=biases,
                learning_rate=self.models[0].lr,
                epochs=self.models[0].epochs)
        

    @classmethod
    def load(cls, filename, input_size, learning_rate=0.001, epochs=100):
        """
        Load model parameters from file and restore model state.
        """

        # Load the saved model parameters from file
        data = np.load(filename)
        weights = data['weights']
        biases = data['biases']

        # Create a new MultiClassPerceptron instance
        n_classes = weights.shape[0]
        model = cls(n_classes, input_size, learning_rate, epochs)

        # Restore the weights and biases for each class
        for i in range(n_classes):
            model.models[i].weights = weights[i]
            model.models[i].bias = biases[i]
        return model

# Widrow-Hoff (Least Mean Squares) Model  
class WidrowHoff:
    """
    Widrow-Hoff model using LMS (least mean squares) learning for multi-class regression/classification.
    """

    def __init__(self, X, T, learning_rate, epochs, variable):
        '''
        Initialize the Widrow-Hoff model.
        '''
        
        self.lr = learning_rate # Set the learning rate
        self.epochs = epochs # Set the number of training epochs
        self.variable = variable # Flag for using learning rate decay

        self.X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add bias column to input
        self.T = T # Target labels

        self.input_size = self.X.shape[1] # Number of input features (including bias)
        self.output_size = T.shape[1] # Number of output classes

        self.weights = np.random.uniform(-0.01, 0.01, size=(self.input_size, self.output_size)) # Initialize weights randomly
        
        print(f"Initialized Widrow-Hoff model with learning rate {self.lr}, epochs {self.epochs}")
        print(f"Input size: {self.input_size}, Output size: {self.output_size}")

    def train(self):
        """
        Train the model using the LMS update rule.
        """

        ib = st.empty()  # create a placeholder outside the loop

        # Train for the specified number of epochs
        for epoch in range(1, self.epochs + 1):
            for x, t in zip(self.X, self.T):
                y = np.dot(x, self.weights) # Compute model output
                error = t - y # Calculate the error
                self.weights += self.lr * np.outer(x, error) # Update weights using LMS rule

            # Periodically update progress and apply learning rate decay if necessary
            if epoch % (self.epochs // 10) == 0 or epoch == 1:

                if self.variable:
                    self.lr *= 0.95  # Apply learning rate decay

                ib.info(f"Epoch {epoch}/{self.epochs} complete")

        ib.info("\nTraining complete!")

    def save(self, filename="model.npz"):
        """
        Save model weights and parameters to a file.
        """

        np.savez(filename,
                 weights=self.weights,
                 learning_rate=self.lr,
                 epochs=self.epochs,
                 input_size=self.input_size,
                 output_size=self.output_size,
                 variable=self.variable)
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename, X, T):
        """
        Load a saved Widrow-Hoff model from file.
        """

        data = np.load(filename)
        model = cls(X, T, data['learning_rate'], int(data['epochs']), data['variable'])
        model.weights = data['weights'] # Restore weights from saved file
        print(f"Model loaded from {filename}")
        return model

    def predict(self, input_vector):
        """
        Predict output vector for a single input vector using the learned weights.
        """

        input_with_bias = np.append(input_vector, 1.0) # Append bias term to input

        return np.dot(input_with_bias, self.weights) # Calculate prediction


# Model Evaluation 
def evaluate(model, X, T, letters):
    """
    Evaluate the model's prediction accuracy over the dataset.
    
    Parameters:
    - model: Trained model object with a predict() method
    - X: Input feature array
    - T: One-hot encoded target labels
    - letters: Class labels (A-Z)
    """

    correct = 0 # Count correct predictions
    total = X.shape[0]  #Define total based on number of samples

    # Loop through all samples in the dataset
    for i in range(total):
        pred_index = model.predict(X[i]) # Get predicted class index
        if isinstance(pred_index, np.ndarray):
            pred_index = pred_index.item() # Convert 0-d array to scalar

        
        actual_index = np.argmax(T[i])  # Get the actual class index from the target labels
        # Retrieve the corresponding letters for predicted and actual classes
        pred_letter = letters[pred_index]
        actual_letter = letters[actual_index]

        #If letters are arrays, convert to string
        if isinstance(pred_letter, (np.ndarray, list)):
            pred_letter = pred_letter.item()
        if isinstance(actual_letter, (np.ndarray, list)):
            actual_letter = actual_letter.item()

        # Check if prediction matches true label
        if pred_index == actual_index:
            correct += 1

    accuracy = correct / total # Compute accuracy as fraction
    print(f"\n Accuracy: {accuracy:.2%}")
