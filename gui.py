import tkinter as tk
from tkinter import ttk
from main import Perceptron, WidrowHoff  # You'll define these
import numpy as np

class ANNApp:
    def __init__(self, root):
        self.root = root
        root.title("ANN Character Classifier")

        # Algorithm selection
        self.algorithm_var = tk.StringVar(value="Perceptron")
        ttk.Label(root, text="Algorithm:").pack()
        ttk.Combobox(root, textvariable=self.algorithm_var, values=["Perceptron", "WidrowHoff"]).pack()

        # Learning rate input
        ttk.Label(root, text="Learning rate (α):").pack()
        self.lr_entry = ttk.Entry(root)
        self.lr_entry.insert(0, "0.1")
        self.lr_entry.pack()

        # Train button
        self.train_btn = ttk.Button(root, text="Train", command=self.train_model)
        self.train_btn.pack(pady=5)

        # Character input (text for simplicity)
        ttk.Label(root, text="Character Input (binary string or image vector):").pack()
        self.char_entry = ttk.Entry(root, width=50)
        self.char_entry.pack()

        # Classify button
        self.classify_btn = ttk.Button(root, text="Classify", command=self.classify_character)
        self.classify_btn.pack(pady=5)

        # Result
        self.result_label = ttk.Label(root, text="Result: ")
        self.result_label.pack(pady=10)

        self.model = None

    def train_model(self):
        algorithm = self.algorithm_var.get()
        learning_rate = float(self.lr_entry.get())
        if algorithm == "Perceptron":
            self.model = Perceptron(learning_rate)
        else:
            self.model = WidrowHoff(learning_rate)
        self.model.train()  # you must implement train() in your classes
        self.result_label.config(text="Model trained using " + algorithm)

    def classify_character(self):
        input_data = self.char_entry.get()
        vector = np.array([int(i) for i in input_data.strip()])
        result = self.model.predict(vector)  # you must implement predict()
        self.result_label.config(text=f"Result: {result}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ANNApp(root)
    root.mainloop()
