import cv2
import numpy as np
import os

# Set your existing directory
output_dir = 'data/inputimages'

# Create the directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

def add_noise(image, noise_type="gaussian", var=0.01):
    """Add noise to an image."""
    if noise_type == "gaussian":
        mean = 0
        sigma = var**0.5
        gaussian = np.random.normal(mean, sigma, image.shape)
        noisy_image = np.clip(image + gaussian, 0, 255).astype(np.uint8)
    elif noise_type == "salt_and_pepper":
        s_vs_p = 0.5
        amount = var
        out = np.copy(image)
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords[0], coords[1]] = 1
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords[0], coords[1]] = 0
        noisy_image = out
    return noisy_image

def save_image(image, filename):
    """Save image to the specified filename."""
    cv2.imwrite(filename, image)

def generate_and_save_images():
    """Generate synthetic images with varying content, color, and noise and save them."""
    index = 1
    for color in ["gray", "red", "green", "blue"]:
        for noise_level in [0, 0.01, 0.05]:
            for size in [(128, 128), (256, 256)]:
                image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
                if color == "gray":
                    image[:] = np.random.randint(0, 256, (size[0], size[1]), dtype=np.uint8)[:, :, None]
                elif color == "red":
                    image[:, :] = [0, 0, 255]
                elif color == "green":
                    image[:, :] = [0, 255, 0]
                elif color == "blue":
                    image[:, :] = [255, 0, 0]
                
                noisy_image = add_noise(image, var=noise_level)
                
                # Define filename and save image
                filename = os.path.join(output_dir, f"{color}_noise_{noise_level}_size_{size}_img_{index}.png")
                save_image(noisy_image, filename)
                
                index += 1

generate_and_save_images()
