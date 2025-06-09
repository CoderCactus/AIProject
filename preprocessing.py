import random

def _add_noise(mat, level):
    """
    Randomly flips a percentage of pixels in a binary matrix.

    Parameters:
    mat: 2D NumPy array with values 0 or 1
    level: float - fraction of pixels to flip

    Returns:
    -Noisy binary matrix
    """
    noisy = mat.copy()
    total = mat.size
    flip_count = int(total * level)

    indices = random.sample(range(total), flip_count)
    for i in indices:
        noisy.flat[i] = 1 - noisy.flat[i]  # flip 0<->1

    return noisy
