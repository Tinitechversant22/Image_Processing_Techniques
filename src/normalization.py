import numpy as np

def pixel_value_scaling(img):
    min_val = np.min(img)
    max_val = np.max(img)
    scaled_img = (img - min_val) / (max_val - min_val)
    return scaled_img

def z_score_normalization(img):
    mean = np.mean(img)
    std = np.std(img)
    normalized_img = (img - mean) / std
    return normalized_img
