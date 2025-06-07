import numpy as np
import cv2
import random

def canvas_to_input_vector(canvas_img,
                           size=(20, 20),
                           add_noise=False,
                           noise_level=0.05):
   
    # Convert RGBA to grayscale
    gray = cv2.cvtColor(canvas_img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)

    # Invert image: white character (1), black background (0)
    gray = cv2.bitwise_not(gray)

    # Resize to 20x20
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

    # Threshold: convert to binary (0 or 1)
    _, binary = cv2.threshold(gray, 128, 1, cv2.THRESH_BINARY)

    # Add noise if enabled
    if add_noise:
        binary = _add_noise(binary, noise_level)

    # Flatten to 1D vector
    return binary.flatten().astype(np.float32)


def _add_noise(mat, level):
    """
    Randomly flips a percentage of pixels in a binary matrix.

    Parameters:
    - mat: 2D NumPy array with values 0 or 1
    - level: float - fraction of pixels to flip

    Returns:
    - Noisy binary matrix
    """
    noisy = mat.copy()
    total = mat.size
    flip_count = int(total * level)

    indices = random.sample(range(total), flip_count)
    for i in indices:
        noisy.flat[i] = 1 - noisy.flat[i]  # flip 0<->1

    return noisy
