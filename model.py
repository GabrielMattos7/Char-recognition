import os
import joblib
import numpy as np
import cv2
print("asduiasgudasguid")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
print("tensorflow XD")
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical
from crop import crop_characters
from build_images import process_all_text_files
from build_dataset import create_label_files

def load_data(train_dir):
    images = []
    labels = []
    for filename in os.listdir(train_dir):
        if filename.endswith('.png'):
            # Load image
            img_path = os.path.join(train_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            print(f"Loaded image {filename} with shape: {img.shape}")
            images.append(img)
            
            # Load corresponding label
            label_path = os.path.join(train_dir, filename.replace('.png', '.label'))
            with open(label_path, 'r') as f:
                label = f.read().strip()
            labels.append(label)
    
    print(f"Total images loaded: {len(images)}")
    print(f"Shape of first image: {images[0].shape}")
    print(f"Shape of last image: {images[-1].shape}")
    
    return images, labels

def augment_image(image):
    """Apply data augmentation to a single image without rotation."""
    augmented_images = [image]
    
    # Add Gaussian noise
    noise = np.random.normal(0, 0.1, image.shape) * 255
    noisy_img = cv2.add(image, noise.astype('uint8'))
    augmented_images.append(noisy_img)
    
    # Shifting image slightly (up, down, left, right)
    for x_shift, y_shift in [(-5, 0), (5, 0), (0, -5), (0, 5)]:
        translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        shifted_img = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
        augmented_images.append(shifted_img)
    
    return augmented_images

def preprocess_data(images, labels):
    # Find the maximum dimensions
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)
    print("MAX HEIGHT: ", max_height)
    print("MAX WIDTH: ", max_width)
    
    augmented_images = []
    augmented_labels = []
    
    # Apply augmentation
    for img, label in zip(images, labels):
        augmented = augment_image(img)
        augmented_images.extend(augmented)
        augmented_labels.extend([label] * len(augmented))
    
    # Pad images to the same size
    padded_images = []
    for img in augmented_images:
        pad_height = max_height - img.shape[0]
        pad_width = max_width - img.shape[1]
        padded_img = np.pad(img, ((0, pad_height), (0, pad_width)), mode='constant')
        padded_images.append(padded_img)
    
    # Convert to numpy array and normalize
    images_array = np.array(padded_images).astype('float32') / 255.0
    
    # Reshape images to flatten them
    images_flat = images_array.reshape(images_array.shape[0], -1)
    
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(augmented_labels)
    labels_onehot = to_categorical(labels_encoded)
    
    return images_flat, labels_onehot, le

def create_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Load data
    train_dir = './cropped_characters'
    images, labels = load_data(train_dir)
    
    # Preprocess data with data augmentation
    X, y, label_encoder = preprocess_data(images, labels)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create and train the model
    input_shape = X_train.shape[1]
    num_classes = y_train.shape[1]

    model = create_model(input_shape, num_classes)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
    
    # Save the trained model
    model.save('character_recognition_model.h5')
    
    # Save the LabelEncoder and scaler
    import joblib
    joblib.dump(label_encoder, 'label_encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Generate classification report
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    class_names = label_encoder.classes_
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names, zero_division=1))

def build_images():
    input_directory = "./text"
    output_directory = "./output_images"
    
    font_size = 30
    text_color = (0, 0, 0)
    bg_color = (255, 255, 255)
    image_size = (1000, 800)

    process_all_text_files(input_directory, output_dir=output_directory, font_size=font_size, text_color=text_color, bg_color=bg_color, image_size=image_size)

def crop():
    input_directory = "./output_images"
    output_directory = "./cropped_characters"
    crop_characters(input_directory, output_directory)

def add_labels():
    text_directory = "./text"
    cropped_directory = "./cropped_characters"
    create_label_files(text_directory, cropped_directory)

if __name__ == "__main__":
    build_images()
    crop()
    add_labels()
    main()
