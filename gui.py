import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2

# Placeholder training functions
def train_perceptron(X, y, learning_rate, epochs=100):
    weights = np.zeros(X.shape[1])
    for _ in range(epochs):
        for xi, target in zip(X, y):
            prediction = np.where(np.dot(xi, weights) >= 0.0, 1, -1)
            weights += learning_rate * (target - prediction) * xi
    return weights

def train_widrow_hoff(X, y, learning_rate, epochs=100):
    weights = np.zeros(X.shape[1])
    for _ in range(epochs):
        for xi, target in zip(X, y):
            output = np.dot(xi, weights)
            weights += learning_rate * (target - output) * xi
    return weights

def classify(weights, input_vec):
    result = np.dot(input_vec, weights)
    return 1 if result >= 0 else -1

# Streamlit UI
st.title("Character Classifier using ANN")
st.sidebar.header("Configuration")

algorithm = st.sidebar.selectbox("Select Algorithm", ["Perceptron", "Widrow-Hoff"])
learning_rate = st.sidebar.slider("Learning Rate (α)", 0.01, 1.0, 0.1, step=0.01)
epochs = st.sidebar.slider("Training Epochs", 10, 500, 100, step=10)

st.header("Training Data")
uploaded_data = st.file_uploader("Upload CSV with Features & Labels", type="csv")

if uploaded_data:
    import pandas as pd
    data = pd.read_csv(uploaded_data)
    st.write(data.head())

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    st.success("Training started...")
    if algorithm == "Perceptron":
        weights = train_perceptron(X, y, learning_rate, epochs)
    else:
        weights = train_widrow_hoff(X, y, learning_rate, epochs)

    st.success(f"Training complete! Weights: {weights}")

    st.header("Test a Character")
    input_vec = st.text_input("Enter character vector (comma-separated)", "1,0,1,1")
    if st.button("Classify"):
        vec = np.array([float(i) for i in input_vec.split(',')])
        result = classify(weights, vec)
        st.write(f"Classification result: {result}")

st.header("Draw a Character")

canvas_result = st_canvas(
    fill_color="white",  # Background color
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data

    st.image(img, caption="Your Drawing", use_container_width=False)

    if st.button("Classify Drawing"):
        # Convert to grayscale, resize to match model input
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (10, 10))  # adjust to your model's expected size
        flattened = resized.flatten() / 255.0  # normalize

        result = classify(weights, flattened)
        st.write(f"Classification result: {result}")