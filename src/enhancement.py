import cv2
import numpy as np

def histogram_equalization(img):
    return cv2.equalizeHist(img)

def adaptive_histogram_equalization(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)
    return cl1

def sharpening(img):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened
