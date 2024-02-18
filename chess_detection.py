import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# gathering data, preprocessing, data augmentation
def preprocess_data(folder_path):
    X = []
    y = []

    # folder labels
    labels = {'pawn': 0, 'bishop': 1, 'knight': 2, 'rook': 3, 'queen': 4}

    # iterate through each folder and each image, preprocess, and add to a numpy array of 
    for label, index in labels.items():
        images = os.listdir(os.path.join(folder_path, label))
        for image_name in images:
            image_path = os.path.join(folder_path, label, image_name)
            image = Image.open(image_path)
            image = image.resize((100, 100))  # resize image to desired dimensions
            image = np.array(image)
            X.append(image)
            y.append(index)

    # add preprocessed images and labels to corresponding numpy arrays
    X = np.array(X)
    y = np.array(y)
    return X, y

# split data
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)