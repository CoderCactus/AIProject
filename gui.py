import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
import cv2
import main
import pandas as pd

def classify(weights, input_vec):
    result = np.dot(input_vec, weights)
    return 1 if result >= 0 else -1

# Streamlit UI
st.title("Character Classifier")
st.sidebar.header("Configuration")

algorithm = st.sidebar.selectbox("Select Algorithm", ["Perceptron", "Widrow-Hoff"])
learning_rate = st.sidebar.slider(
    "Learning rate",
    min_value=0.001,
    max_value=0.01,
    value=0.005,      # Default value shown on the slider
    step=0.001,      # Smallest increment
    format="%.4f"      # Display with 5 decimal places
)
epochs = st.sidebar.slider("Training Epochs", 5000, 40000, 20000, step=2000)

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
    stroke_color="white",
    background_color="black",
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

            st.session_state["last_output"] = output

            st.info(f"Classification Result: {result}")

if "last_output" in st.session_state:
    if st.button("Output Vector"):
        st.snow()
        out = pd.DataFrame(
            [np.round(st.session_state.last_output, 3)],
            columns=main.letters_list
        )
        st.dataframe(out)
            


