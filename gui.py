import streamlit as st # Web interface framework for interactive apps
import numpy as np # For numerical computations and arrays
from streamlit_drawable_canvas import st_canvas # Interactive canvas widget for drawing
import cv2 # OpenCV for image processing tasks
import main # Custom module where models, data, and utility functions live
import pandas as pd # For handling and displaying tabular data

# Simple binary classifier using weights
def classify(weights, input_vec):
    """
    Classify an input vector by computing the dot product with the weight vector.
    Returns 1 if the result is non-negative (indicating 'positive class'), otherwise -1.
    """
    result = np.dot(input_vec, weights)
    return 1 if result >= 0 else -1

# Streamlit UI
st.title("Character Classifier") # App title shown at the top
st.sidebar.header("Configuration") # Sidebar section for configuring training settings

# Sidebar: algorithm selection
algorithm = st.sidebar.selectbox("Select Algorithm", ["Perceptron", "Widrow-Hoff"]) # Dropdown menu for choosing classification algorithm
# Sidebar: learning rate slider
learning_rate = st.sidebar.slider(
    "Learning rate",
    min_value=0.001, # Minimum selectable learning rate
    max_value=0.01, # Maximum selectable learning rate
    value=0.005, # Default value shown on the slider
    step=0.001, # Step size between options
    format="%.3f" # Show 3 decimal places in UI

# Sidebar: optional variable learning rate
variable = st.sidebar.toggle('Variable Learning Rate', help = "Only works for Widrow-Hoff") # Only applies to Widrow-Hoff. Allows learning rate to decay gradually across epochs

# Sidebar: training epochs as powers of 10
epochs = 10**(st.sidebar.slider("Training Epochs (10^)", 1, 5, 2, step=1))

# Load a Previously Saved Model from File
if st.sidebar.button("Load Saved Model"):  # Load a saved model from file 
    try:
        if algorithm == "Widrow-Hoff":
        # Load Widrow-Hoff model with stored weights and settings
            model = main.WidrowHoff.load(
                f"Models/model_{algorithm}_lr{learning_rate}_ep{epochs}_variable{variable}.npz",
                main.X,
                main.T,
                variable
            )
        else:
        # Load Perceptron model with stored settings
            n_classes = main.T.shape[1]
            input_size = main.X.shape[1]
            model = main.MultiClassPerceptron.load(
                f"Models/model_{algorithm}_lr{learning_rate}_ep{epochs}_variable{variable}.npz",
                input_size,
                learning_rate,
                epochs
            )
        # Save model instance to Streamlit's session state for reuse
        st.session_state.model = model
        st.success("Model loaded from disk")

    except FileNotFoundError:
    # Display error message if file is missing
        st.error("No saved model found")

# Train model based on selected algorithm
if st.button("Train Model"):
    if algorithm == "Perceptron":

        # Shuffle data to ensure better generalization during training
        indices = np.random.permutation(len(main.X))
        X_shuffled = main.X[indices]
        T_shuffled = main.T[indices]

        # Determine number of classes and input features
        n_classes = T_shuffled.shape[1]
        input_size = X_shuffled.shape[1]
        # Create and train the model
        model = main.MultiClassPerceptron(n_classes, input_size, learning_rate, epochs, variable)
        model.train(X_shuffled, T_shuffled)
        st.session_state.model = model

    elif algorithm == "Widrow-Hoff":
        # Widrow-Hoff: train once for all classes
        model = main.WidrowHoff(main.X, main.T, learning_rate, epochs, variable)
        model.train()
        st.session_state.model = model

# Drawing Interface
st.header("Draw a Character")

# Draw a character on canvas (user draws with mouse/finger)
canvas_result = st_canvas(
    fill_color="white", # Color used to fill inside shapes
    stroke_width=15, # Thickness of drawn strokes
    stroke_color="white", # Drawing color (white on black background)
    background_color="black", # Canvas background color (blackboard style)
    height=280, # Canvas height in pixels
    width=280, # Canvas width in pixels
    drawing_mode="freedraw", # Allow freehand drawing
    key="canvas", # Unique identifier for widget state
)

# Predict Drawing
if canvas_result.image_data is not None:
    img = canvas_result.image_data # Get the drawing from the canvas

    if st.button("Classify Drawing"):
        if "model" not in st.session_state:
            st.warning("Train the model first.")  # Show a warning if the model is not trained yet
        else:
            img_array = np.array(img) # Convert the canvas image to a NumPy array
            gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY) # Convert to grayscale
            resized = cv2.resize(gray, (20, 20))  # Rescale to 20x20 pixels
            flattened = resized.flatten() / 255.0  # Normalize to [0, 1] range
            # Model Prediction
            model = st.session_state.model
            if algorithm == "Widrow-Hoff":
                output = model.predict(flattened)             # Raw output vector
                predicted_index = np.argmax(output)           # Choose class with highest score
                st.session_state["last_output"] = output      # Store output for display
            else:  # Perceptron prediction returns index
                predicted_index = model.predict(flattened)[0] 

            # Display result as letter (A-Z)
            result = main.letters_list[predicted_index]
            st.info(f"Classification Result: {result}")

# Show Output Vector (Only for Widrow-Hoff)
if "last_output" in st.session_state and algorithm == "Widrow-Hoff":
    if st.button("Output Vector"):
        st.snow() # Fun visual animation
        out = pd.DataFrame(
            [np.round(st.session_state.last_output, 3)],
            columns=main.letters_list
        )
        st.dataframe(out) # Display confidence scores for each letter class

# Save Trained Model
if st.sidebar.button("Save Model"):
    if "model" in st.session_state:
        filename = f"Models/model_{algorithm}_lr{learning_rate}_ep{epochs}_variable{variable}.npz"
        st.session_state.model.save(filename) # Save weights and settings to .npz file
        st.success(f"Model saved at {filename}")
    else:
        st.warning("No model in memory to save.") # Show warning if model isn’t trained