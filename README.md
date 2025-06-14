# Character Classifier

This is a Streamlit-based web app for training and testing character classification models using either the **Perceptron** or **Widrow-Hoff** learning algorithms. Users can draw letters, train models, load/save them, and get predictions in real-time.


## Features

* **Interactive Canvas**: Draw handwritten characters (A–Z) directly in the browser.
* **Training Options**: Choose between Perceptron or Widrow-Hoff algorithm.
* **Custom Configurations**:

  * Set learning rate
  * Choose fixed or variable learning rate (Widrow-Hoff only)
  * Define number of epochs (10^n)
* **Model Persistence**: Save and load models from disk.
* **Output Inspection**: View output vectors (Widrow-Hoff).
* **Multi-class Support**: Supports full A–Z letter classification.


## Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

**Required Libraries:**

* `streamlit`
* `numpy`
* `opencv-python`
* `pandas`
* `streamlit-drawable-canvas`


## Getting Started

1. Launch the Streamlit app:

   ```bash
   streamlit run gui.py
   ```

2. Open the web interface in your browser, draw a character, train a model and classify.


## Algorithms Implemented

### 1. **Perceptron**

* One-vs-all strategy for multi-class classification
* Fixed learning rate
* Binary and Multi-class support

### 2. **Widrow-Hoff rule**

* Batch weight updates
* Optional variable learning rate (decays over time)


## Dataset

The training dataset is a `.npz` file:

* 26 classes (A–Z)
* Each image: 20x20 black and white
* Normalized pixel values (0–1)
* One-hot encoded labels


## Model Saving Format

Each model is saved with:

* Weights and biases
* Training parameters (learning rate, epochs)
* File naming based on selected configuration:

  ```
  Models/model_<Algorithm>_lr<rate>_ep<epochs>_variable<True/False>.npz
  ```


## Example Use

* Train a model with `Widrow-Hoff`, learning rate `0.005`, and `1000` epochs.
* Draw the letter "A" and classify.
* View the 26-dimensional output vector.
* Save the model.
* Reload it later and test with new drawings.


## Author

* Developed by Constança Rocha, Maiara Almada, Neda Razavi and Tiago Silveira

