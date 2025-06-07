import numpy as np
import cv2
import random

def canvas_to_input_vector(canvas_img,
                           size=(20, 20),
                           add_noise=False,
                           noise_level=0.05):
    gray = cv2.cvtColor(canvas_img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    gray = cv2.bitwise_not(gray)
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(gray, 128, 1, cv2.THRESH_BINARY)

    if add_noise:
        binary = _add_noise(binary, noise_level)

    return binary.flatten().astype(np.float32)

def _add_noise(mat, level):
    noisy = mat.copy()
    total = mat.size
    flip = int(total * level)
    indices = random.sample(range(total), flip)
    for i in indices:
        noisy.flat[i] = 1 - noisy.flat[i]
    return noisy
