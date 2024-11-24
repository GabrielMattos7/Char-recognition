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
    current_height, current_width = image.shape[:2]
    
    if current_height > max_height or current_width > max_width:
        processed_img = cv2.resize(image, (max_width, max_height), interpolation=cv2.INTER_AREA)
    else:
        pad_height = max_height - current_height
        pad_width = max_width - current_width
        processed_img = np.pad(image, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

    normalized_img = processed_img.astype('float32') / 255.0
    flattened_img = normalized_img.flatten().reshape(1, -1)
    
    return flattened_img

def predict_character(image_path, scaler, model, label_encoder):
    new_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    preprocessed_image = preprocess_image(new_image, 32, 30)
    standardized_image = scaler.transform(preprocessed_image)
    prediction = model.predict(standardized_image)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    return predicted_label[0]

def infer_text():
    create_inference_image_from_text()
    bounding_boxes = crop_inference_image()

    model = load_model('character_recognition_model.h5')
    label_encoder = joblib.load('label_encoder.pkl')
    scaler = joblib.load('scaler.pkl')

    inference_dir = "./inference"
    predicted_chars = []

    spacing_threshold = 1 
    previous_x = 0
    previous_w = 0
    previous_char = ''

    image_files = sorted([f for f in os.listdir(inference_dir) if '_' in f and f.endswith('.png')])

    for image_file, (x, w, _, _) in zip(image_files, bounding_boxes):
        distance = max(0, x - (previous_x + previous_w))
        print(f"Current x: {x}, Previous x+w: {previous_x + previous_w}, Distance: {distance}")
        if distance >= spacing_threshold:
            predicted_chars.append(" ") 
        
        image_path = os.path.join(inference_dir, image_file)
        predicted_char = predict_character(image_path, scaler, model, label_encoder)
        print(f"PREVIOUS CHAR {previous_char} PREDICTED: {predicted_char}")
        predicted_chars.append(predicted_char)
        
        previous_char = predicted_char
        previous_x = x
        previous_w = w

    predicted_string = ''.join(predicted_chars)

    print(f"Predicted string: {predicted_string}")

    for image_file in image_files:
        image_path = os.path.join(inference_dir, image_file)
        os.remove(image_path)

if __name__ == "__main__":
    infer_text()

