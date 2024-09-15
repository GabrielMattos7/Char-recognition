import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

def load_data(train_dir):
    images = []
    labels = []
    for filename in os.listdir(train_dir):
        if filename.endswith('.png'):
            # Load image
            img_path = os.path.join(train_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            images.append(img)
            
            # Load corresponding label
            label_path = os.path.join(train_dir, filename.replace('.png', '.label'))
            with open(label_path, 'r') as f:
                label = f.read().strip()
            labels.append(label)
    
    return np.array(images), np.array(labels)

def preprocess_data(images, labels):
    # Normalize images
    images = images.astype('float32') / 255.0
    
    # Reshape images to flatten them
    images = images.reshape(images.shape[0], -1)
    
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    labels_onehot = to_categorical(labels_encoded)
    
    return images, labels_onehot, le

def create_model(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Load data
    train_dir = './train'
    images, labels = load_data(train_dir)
    
    # Preprocess data
    X, y, label_encoder = preprocess_data(images, labels)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    input_shape = X_train.shape[1]
    num_classes = y_train.shape[1]
    model = create_model(input_shape, num_classes)
    
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Generate classification report
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    class_names = label_encoder.classes_
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

if __name__ == "__main__":
    main()
