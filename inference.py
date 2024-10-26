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
    crop_characters(directory, directory)

model = load_model('character_recognition_model.h5')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

def preprocess_image(image, max_height, max_width):
    pad_height = max_height - image.shape[0]
    pad_width = max_width - image.shape[1]
    padded_img = np.pad(image, ((0, pad_height), (0, pad_width)), mode='constant')

    normalized_img = padded_img.astype('float32') / 255.0
    
    flattened_img = normalized_img.flatten().reshape(1, -1)
    
    return flattened_img

create_inference_image_from_text()
crop_inference_image()

def predict_character(image_path):
    new_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    preprocessed_image = preprocess_image(new_image, 28, 17)
    standardized_image = scaler.transform(preprocessed_image)
    prediction = model.predict(standardized_image)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    return predicted_label[0]

inference_dir = "./inference"
predicted_chars = []

image_files = sorted([f for f in os.listdir(inference_dir) if '_' in f and f.endswith('.png')])

for image_file in image_files:
    image_path = os.path.join(inference_dir, image_file)
    predicted_char = predict_character(image_path)
    predicted_chars.append(predicted_char)

predicted_string = ''.join(predicted_chars)

print(f"Predicted string: {predicted_string}")

for image_file in image_files:
    image_path = os.path.join(inference_dir, image_file)
    os.remove(image_path)

