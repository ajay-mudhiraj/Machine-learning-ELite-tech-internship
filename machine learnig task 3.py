# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 23:03:27 2025

@author: Ajay
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define paths (replace these with your actual dataset paths)
CAT_PATH = "./data/cats"
DOG_PATH = "./data/dogs"
IMG_SIZE = (64, 64)  # Resize images to 64x64 for consistency

# Load images and labels
def load_images_labels(path, label):
    images = []
    labels = []
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        try:
            img = load_img(img_path, target_size=IMG_SIZE)
            img_array = img_to_array(img) / 255.0  # Normalize pixel values
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels)

# Load cats and dogs data
cat_images, cat_labels = load_images_labels(CAT_PATH, 0)  # Label cats as 0
dog_images, dog_labels = load_images_labels(DOG_PATH, 1)  # Label dogs as 1

# Combine data
images = np.concatenate((cat_images, dog_images), axis=0)
labels = np.concatenate((cat_labels, dog_labels), axis=0)

# Flatten images for SVM
images_flat = images.reshape(images.shape[0], -1)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images_flat, labels, test_size=0.2, random_state=42)

# Train SVM model
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# Predict on test data
y_pred = svm.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Test on a new image
def predict_image(img_path):
    try:
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_flat = img_array.reshape(1, -1)
        prediction = svm.predict(img_flat)
        return "Dog" if prediction[0] == 1 else "Cat"
    except Exception as e:
        print(f"Error predicting image {img_path}: {e}")

# Example: Predict a new image
new_image_path = "./data/test_image.jpg"
print("Prediction for new image:", predict_image(new_image_path))
