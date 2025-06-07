# perceptron.py

import numpy as np  # Import NumPy for numerical operations

def unit_step_func(x):
    return np.where(x > 0, 1, 0) #defining the activation function: if x is bigger than 0 retun 1, if not return 0

class Perceptron:
    def __init__(self,learning_rate=0.01, n_inters=1000):
        """
        Initializes the perceptron model with:
        - learning_rate: how much weights are updated
        - n_iterates: how many times to iterate over the dataset
        """
        self.weights = None # Initialize weights to 0
        self.bias =None                   # Initialize bias to 0
        self.lr = learning_rate              # Store learning rate
        self.n_iters= n_iters                 # Store number of iterations
        self.activation_func= unit_step_func  


    #for the trainning data
   def fit(self,X, y):
        """
        Trains the perceptron using the Perceptron learning rule
        - n_samples, n_features: rows and columns of the matrix of input samples
        - y_train: expected outputs
        """
       n_samples, n_features= X.shape
       #init parameters
        self.wieghts=np.zeros(n_features)
        self.bias=0
        y_ =np.where(y>0, 1, 0) #making sure that the class lables are 1 and 0 instead of 1 and -1 for example
    
        #learn weights 
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):# Loop through each training sample
                linear_output=np.dot(x_i, self.weights) + self.bias # Weighted sum
                y_predicted=self.activation_func(linear_output) # Predict output

                #perceptron update rule
                update = self.lr*(y_[idx]-y_predicted)
                self.weights += update * x_i
                self.bias += update
                

    #test data
    def predict(self, x):
        """
        Predicts the output (0 or 1) for a single input vector `x`
        """
        linear_output = np.dot(self.weights, x) + self.bias  # Weighted sum
        return self.activation_func(linear_output)                # Apply step function

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

