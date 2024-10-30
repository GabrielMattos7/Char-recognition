from tensorflow.keras.models import load_model
import joblib
import cv2
import numpy as np
from build_images import create_text_image 
from crop import crop_characters
import os

def create_inference_image_from_text():
    file = "./inference/text.txt"
    output_file = "./inference/text.png"
    font_size = 30
    image_size = (1000, 800)
    create_text_image(file, output_file, font_size, image_size=image_size)

def crop_inference_image():
    directory = "./inference"
    bounding_boxes = crop_characters(directory, directory)
    return bounding_boxes

def preprocess_image(image, max_height, max_width):
    # Get the current height and width of the image
    current_height, current_width = image.shape[:2]
    
    # Check if padding or resizing is needed
    if current_height > max_height or current_width > max_width:
        # Resize the image if it's larger than the target size
        processed_img = cv2.resize(image, (max_width, max_height), interpolation=cv2.INTER_AREA)
    else:
        # Calculate the padding if the image is smaller
        pad_height = max_height - current_height
        pad_width = max_width - current_width

        # Apply padding (if necessary)
        processed_img = np.pad(image, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

    # Normalize the image
    normalized_img = processed_img.astype('float32') / 255.0
    
    # Flatten the image and reshape it into a single row
    flattened_img = normalized_img.flatten().reshape(1, -1)
    
    return flattened_img

create_inference_image_from_text()
aaa = crop_inference_image()

def predict_character(image_path):
    new_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    preprocessed_image = preprocess_image(new_image, 32, 32)
    standardized_image = scaler.transform(preprocessed_image)
    prediction = model.predict(standardized_image)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    return predicted_label[0]


model = load_model('character_recognition_model.h5')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')


inference_dir = "./inference"
predicted_chars = []

spacing_threshold = 9  # Adjust as needed for your font/spacing
predicted_chars = []
previous_horizontal = 0


image_files = sorted([f for f in os.listdir(inference_dir) if '_' in f and f.endswith('.png')])
# print(len(image_files))
# print(len(aaa))
for image_file, (x,w,_,_) in zip(image_files, aaa):
    # distance = abs(x - previous_horizontal)  # Using abs to avoid negative values
    # print(distance)
    # 
    # if distance >= spacing_threshold:
        # predicted_chars.append(" ")  # Add space when distance exceeds threshold
    # previous_horizontal = x + w
    predicted_chars.append(" ")
    image_path = os.path.join(inference_dir, image_file)
    predicted_char = predict_character(image_path)
    predicted_chars.append(predicted_char)

predicted_string = ''.join(predicted_chars)

print(f"Predicted string: {predicted_string}")

for image_file in image_files:
    image_path = os.path.join(inference_dir, image_file)
    os.remove(image_path)
