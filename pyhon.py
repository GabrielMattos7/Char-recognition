import cv2
import numpy as np

# Load the image in grayscale
img = cv2.imread("print.png", cv2.IMREAD_GRAYSCALE)

# Apply binary thresholding to convert the image to black and white
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

# Find contours in the binary image
contours, _ = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours from left to right (based on the x-coordinate of the bounding box)
contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

# Create a list to hold images of each digit
digit_images = []

# Loop through each contour and extract the digit
for i, contour in enumerate(contours):
    # Get bounding box for the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Extract the digit from the original image using the bounding box
    digit_img = img[y:y+h, x:x+w]
    
    # Optionally resize the digit image for consistent display
    # digit_img = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_NEAREST)
    
    # Add the digit image to the list
    digit_images.append(digit_img)
    
    # Display the digit image
    cv2.imshow(f"Digit {i+1}", digit_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    
    # Optionally, save the digit image to disk
    # cv2.imwrite(f"digit_{i+1}.png", digit_img)

# Wait for any key press to close the windows
print(len(contours))
