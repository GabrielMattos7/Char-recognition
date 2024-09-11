import cv2
import numpy as np
#TODO: offseat analysis to find spaces between words
#TODO: create MLP model 
#TODO: find dataset for the chars
#TODO: train (jetson?)
#TODO?: implement explainable AI
# Load the image in grayscale

#pseudo for first todo
# def check_for_gaps(contours):
    # last_w = -1
    # last_x2 = -1
    # last_y = -1
    # maan_w = -1
    # mean_y = -1
    # contour = contours[0]
    # for contour in contours:
        # x,_,w,_ = cv2.boundingRect(contour)
        # x2 = x + w
        # if((x - last_x2) > mean_w):
            # print("theres a space.")
        # elif((y - last_y) > mean_y):
            # print("theres a new line")
        # last_w = w
        # last_x2 = x2
        # last_y


show = False
img = cv2.imread("img.png", cv2.IMREAD_GRAYSCALE)
# aaa
# a a
# Apply binary thresholding to convert the image to black and white
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Find contours in the binary image
contours, _ = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours from left to right (based on the x-coordinate of the bounding box)
contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

# Create a list to hold images of each digit
digit_images = []

# Loop through each contour and extract the digit
if show:
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
img_aux = np.zeros_like(img)
print(img_aux.shape)
cv2.drawContours(img_aux, contours, -1, (255, 255, 255), 1)
# Wait for any key press to close the windows
print(len(contours))
cv2.imwrite("test_cv.THRESH_BINARY_INV+cv.THRESH_OTSU.png",img_aux)
cv2.imshow("img", img_aux)
cv2.waitKey(0)
cv2.destroyAllWindows
