import os
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from keras.applications import VGG16
import matplotlib.pyplot as plt
import seaborn as sns

# gathering data, preprocessing, data augmentation
def preprocess_data(folder_path):
    image_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    X = []
    y = []

    # folder labels
    labels = {'pawn': 0, 'bishop': 1, 'knight': 2, 'rook': 3, 'queen': 4}

    # iterate through each folder and each image, preprocess, and add to numpy arrays
    for label, index in labels.items():
        images = os.listdir(os.path.join(folder_path, label))
        for image_name in images:
            image_path = os.path.join(folder_path, label, image_name)
            image = Image.open(image_path)
            image = image.resize((100, 100)) 
            image = np.array(image)
            image = image_datagen.random_transform(image) # preprocessing step
            X.append(image)
            y.append(index)

    # add preprocessed images and labels to corresponding numpy arrays
    X = np.array(X)
    y = np.array(y)
    return X, y

# split data
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

# simple random forest model
def train_random_forest(X_train, X_test, y_train, y_test):
    rf_classifier = RandomForestClassifier(n_estimators=10, random_state=42)
    rf_classifier.fit(X_train.reshape(len(X_train), -1), y_train)
    
    train_accuracy = rf_classifier.score(X_train.reshape(len(X_train), -1), y_train)
    test_accuracy = rf_classifier.score(X_test.reshape(len(X_test), -1), y_test)
    
    return rf_classifier, train_accuracy, test_accuracy

# main
if __name__ == "__main__":
    folder_path = "images"  
    X, y = preprocess_data(folder_path)

    X_train, X_test, y_train, y_test = split_data(X, y)

    # train simple rf model
    rf_model, train_accuracy, test_accuracy = train_random_forest(X_train, X_test, y_train, y_test)

    rf_y_pred = rf_model.predict(X_test.reshape(len(X_test), -1))
    conf_matrix = confusion_matrix(y_test, rf_y_pred)

    print("Random Forest Train Accuracy:", train_accuracy)
    print("Random Forest Test Accuracy:", test_accuracy)
    print(conf_matrix)

    # plot confusion matrix using seaborn
    labels = ['pawn', 'bishop', 'knight', 'rook', 'queen']
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix_rf.png")  # save image step
    plt.show()
