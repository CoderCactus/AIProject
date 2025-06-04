import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2
import main

def classify(weights, input_vec):
    result = np.dot(input_vec, weights)
    return 1 if result >= 0 else -1

# Streamlit UI
st.title("Character Classifier")
st.sidebar.header("Configuration")

algorithm = st.sidebar.selectbox("Select Algorithm", ["Perceptron", "Widrow-Hoff"])
learning_rate = st.sidebar.slider(
    "Training Epochs",
    min_value=0.000001,
    max_value=0.01,
    value=0.00001,      # Default value shown on the slider
    step=0.0001,      # Smallest increment
    format="%.6f"      # Display with 5 decimal places
)
epochs = st.sidebar.slider("Training Epochs", 5000, 100000, 20000, step=1000)

if st.button("Train Model"):
    if algorithm == "Perceptron":
        #to be written
        print()
    elif algorithm == "Widrow-Hoff":
        model = main.WidrowHoff(main.X, main.T, learning_rate, epochs)
        model.train()
        st.session_state.model = model

st.header("Draw a Character")

canvas_result = st_canvas(
    fill_color="white",  # Background color
    stroke_width=10,
    stroke_color="orange",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data

    if st.button("Classify Drawing"):
        if "model" not in st.session_state:
            st.warning("Train the model first.")
        else:
            # Convert to grayscale, resize to match model input
            img_array = np.array(img)
            gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (20, 20))  # adjust to your model's expected size
            flattened = resized.flatten() / 255.0  # normalize

            model = st.session_state.model
            output = model.predict(flattened)
            predicted_index = np.argmax(output)
            result = main.letters_list[predicted_index]
            st.write(f"Classification Result: {result}")


    