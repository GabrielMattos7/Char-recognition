import cv2
import numpy as np
import os

def sort_contours(contours, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    (contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes),
        key=lambda b:b[1][i], reverse=reverse))
    return contours, bounding_boxes

def crop_characters(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours from left to right, then top to bottom
            (contours, _) = sort_contours(contours, "top-to-bottom")
            
            # Get the corresponding text
            text = os.path.splitext(filename)[0]
            
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                
                # Add some padding
                padding = 2
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(img.shape[1] - x, w + 2*padding)
                h = min(img.shape[0] - y, h + 2*padding)
                
                char_image = img[y:y+h, x:x+w]
                output_filename = f"{os.path.splitext(filename)[0]}_{i:03d}.png"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, char_image)
                print(f"Saved {output_filename}")

if __name__ == "__main__":
    input_directory = "./output_images"
    output_directory = "./cropped_characters"
    crop_characters(input_directory, output_directory)
