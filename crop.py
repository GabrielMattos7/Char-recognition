import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def split_characters(image_path, text_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to preprocess the image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Calculate vertical projection profile
    vertical_projection = np.sum(thresh, axis=0)

    # Find word boundaries
    word_boundaries = []
    in_word = False
    word_start = 0
    space_threshold = 10  # Adjust this value to fine-tune space detection

    for i, proj in enumerate(vertical_projection):
        if not in_word and proj > 0:
            word_start = i
            in_word = True
        elif in_word and proj == 0:
            if i - word_start > space_threshold:
                word_boundaries.append((word_start, i))
            in_word = False

    if in_word:
        word_boundaries.append((word_start, len(vertical_projection)))

    char_images = []
    char_positions = []

    for word_start, word_end in word_boundaries:
        word_image = thresh[:, word_start:word_end]
        
        # Find contours in the word image
        contours, _ = cv2.findContours(word_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours from left to right
        contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

        for contour in contours:
            # Get bounding box for each contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out very small contours
            if w * h < 100:  # Adjust this threshold as needed
                continue
            
            # Extract the character from the word image
            char = word_image[y:y+h, x:x+w]
            
            # Add padding
            char = cv2.copyMakeBorder(char, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            
            # Resize to a fixed size (optional)
            char = cv2.resize(char, (50, 50))
            
            char_images.append(char)
            char_positions.append(word_start + x)
        
        # Add a blank image to represent space between words
        if word_end < thresh.shape[1]:
            space = np.zeros((50, 50), dtype=np.uint8)
            char_images.append(space)
            char_positions.append(word_end)

    # Read the corresponding text file
    with open(text_path, 'r') as f:
        text = f.read().strip()

    return char_images, char_positions, text

def save_characters(char_images, char_positions, text, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    text_index = 0
    for i, (char_img, position) in enumerate(zip(char_images, char_positions)):
        if np.sum(char_img) == 0:  # This is a space
            label = ' '
            text_index += 1
        else:
            label = text[text_index]
            text_index += 1

        # Save image
        img_path = os.path.join(output_dir, f"{i:03d}.png")
        cv2.imwrite(img_path, char_img)

        # Save label
        label_path = os.path.join(output_dir, f"{i:03d}.label")
        with open(label_path, 'w') as f:
            f.write(label)

def plot_characters(char_images):
    # Calculate the total width of the image
    total_width = len(char_images) * 50  # Each character is 50 pixels wide
    height = 50  # Height of each character

    # Create a new image to hold all characters
    combined_image = np.zeros((height, total_width), dtype=np.uint8)

    # Place each character in the combined image
    for i, char in enumerate(char_images):
        combined_image[0:height, i*50:(i+1)*50] = char

    # Plot the combined image
    plt.figure(figsize=(total_width/50, 2))  # Adjust figure size
    plt.imshow(combined_image, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    image_path = "./output_images/001.png"
    text_path = "./text/001.txt"
    output_dir = "./train"

    char_images, char_positions, text = split_characters(image_path, text_path)
    save_characters(char_images, char_positions, text, output_dir)
    plot_characters(char_images)

    print(f"Characters and labels saved in {output_dir}")
