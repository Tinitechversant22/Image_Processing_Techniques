import cv2

def rgb_to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def rgb_to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def rgb_to_ycbcr(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
