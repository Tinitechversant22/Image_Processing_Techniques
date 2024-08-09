import cv2
import os
from noise_reduction import gaussian_blur, median_filter
from normalization import pixel_value_scaling
from color_space import rgb_to_grayscale
from enhancement import histogram_equalization

def main():
    image_path = os.path.join("data", "inputimages", "green_noise_0_size_(256, 256)_img_14.png")
    image_path1 = os.path.join("data","inputimages","edgedetect.jpg")
    output_dir = os.path.join("data", "processedimages")

    original_img = cv2.imread(image_path)
    original_img_edge = cv2.imread(image_path1)
    # Convert to grayscale
    gray_img = rgb_to_grayscale(original_img)
    gray_img_edge = rgb_to_grayscale(original_img_edge)

    # Enhancement
    enhanced_img = histogram_equalization(gray_img)

    # Noise reduction
    gaussian_blur_img = gaussian_blur(original_img_edge)
    median_filter_img = median_filter(original_img_edge)

    # Save processed images
    cv2.imwrite(os.path.join(output_dir, "grayscale_img.jpg"), gray_img)
    cv2.imwrite(os.path.join(output_dir, "enhanced_img.jpg"), enhanced_img)
    cv2.imwrite(os.path.join(output_dir, "gaussian_blur_img.jpg"), gaussian_blur_img)
    cv2.imwrite(os.path.join(output_dir, "median_filter_img.jpg"), median_filter_img)

if __name__ == "__main__":
    main()
