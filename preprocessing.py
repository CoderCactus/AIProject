import numpy as np
import cv2

def canvas_to_input_vector(canvas_img, size=(20, 20)):
    gray = cv2.cvtColor(canvas_img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    gray = cv2.bitwise_not(gray)
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(gray, 128, 1, cv2.THRESH_BINARY)
    return binary.flatten().astype(np.float32)
