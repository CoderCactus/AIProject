# Import required libraries
import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
import cv2
import main
import pandas as pd

# Simple binary classifier using weights
def classify(weights, input_vec):
    result = np.dot(input_vec, weights)
    return 1 if result >= 0 else -1

# ----Streamlit UI----
# Set page title
st.title("Character Classifier")
st.sidebar.header("Configuration")

# Sidebar: choose algorithm
algorithm = st.sidebar.selectbox("Select Algorithm", ["Perceptron", "Widrow-Hoff"])
# Sidebar: learning rate slider
learning_rate = st.sidebar.slider(
    "Learning rate",
    min_value=0.001,
    max_value=0.01,
    value=0.005,      # Default value shown on the slider
    step=0.001,      # Smallest increment
    format="%.3f"      # Display with 5 decimal places
)

# Sidebar: optional variable learning rate
variable = st.sidebar.toggle('Variable Learning Rate', help = "Only works for Widrow-Hoff")

# Sidebar: training epochs as powers of 10
epochs = 10**(st.sidebar.slider("Training Epochs (10^)", 1, 5, 2, step=1))

if st.sidebar.button("Load Saved Model"):  # Load a saved model from file 
    try:
        if algorithm == "Widrow-Hoff":
            model = main.WidrowHoff.load(
                f"Models/model_{algorithm}_lr{learning_rate}_ep{epochs}_variable{variable}.npz",
                main.X,
                main.T,
                variable
            )
        else:
            n_classes = main.T.shape[1]
            input_size = main.X.shape[1]
            model = main.MultiClassPerceptron.load(
                f"Models/model_{algorithm}_lr{learning_rate}_ep{epochs}_variable{variable}.npz",
                input_size,
                learning_rate,
                epochs
            )

        st.session_state.model = model
        st.success("Model loaded from disk")

    except FileNotFoundError:
        st.error("No saved model found")

# Train model based on selected algorithm
if st.button("Train Model"):
    if algorithm == "Perceptron":

        # Perceptron: train one for each class (one-vs-all)
        indices = np.random.permutation(len(main.X))
        X_shuffled = main.X[indices]
        T_shuffled = main.T[indices]


        n_classes = T_shuffled.shape[1]
        input_size = X_shuffled.shape[1]
    
        model = main.MultiClassPerceptron(n_classes, input_size, learning_rate, epochs, variable)
        model.train(X_shuffled, T_shuffled)

        st.session_state.model = model
    elif algorithm == "Widrow-Hoff":
        # Widrow-Hoff: train once for all classes
        model = main.WidrowHoff(main.X, main.T, learning_rate, epochs, variable)
        model.train()
        st.session_state.model = model

st.header("Draw a Character")

# Draw a character on canvas (280x280 pixels)
canvas_result = st_canvas(
    fill_color="white",  # Background color
    stroke_width=15,
    stroke_color="white",
    background_color="black", # Canvas background
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# ----Predict Drawing----
if canvas_result.image_data is not None:
    img = canvas_result.image_data # Get the drawing from the canvas

    if st.button("Classify Drawing"):
        if "model" not in st.session_state:
            st.warning("Train the model first.")  # Show a warning if the model is not trained yet
        else:
            img_array = np.array(img) # Convert the canvas image to a NumPy array
            gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY) # Convert to grayscale
            resized = cv2.resize(gray, (20, 20))  # Resize to 20x20 pixels
            flattened = resized.flatten() / 255.0  # normalize

            model = st.session_state.model
            if algorithm == "Widrow-Hoff":
                output = model.predict(flattened)             # ➜ returns vector of shape (26,)
                predicted_index = np.argmax(output)           # ➜ pick class with max score
                st.session_state["last_output"] = output      # ➜ save vector for inspection
            else:  # Perceptron
                predicted_index = model.predict(flattened)[0] # ➜ returns int index

            result = main.letters_list[predicted_index]
            st.info(f"Classification Result: {result}")
# ----Show Output Vector----
if "last_output" in st.session_state and algorithm == "Widrow-Hoff":
    if st.button("Output Vector"):
        st.snow()
        out = pd.DataFrame(
            [np.round(st.session_state.last_output, 3)],
            columns=main.letters_list
        )
        st.dataframe(out)
# ----Save Trained Model---- 
# Save current model to file
if st.sidebar.button("Save Model"):
    if "model" in st.session_state:
        filename = f"Models/model_{algorithm}_lr{learning_rate}_ep{epochs}_variable{variable}.npz"
        st.session_state.model.save(filename)
        st.success(f"Model saved at {filename}")
    else:
        st.warning("No model in memory to save.")
