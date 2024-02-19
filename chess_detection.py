import os
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.applications import VGG16
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model

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
            image = image.resize((224, 224)) 
            image = np.array(image)
            image = image_datagen.random_transform(image) # preprocessing step
            X.append(image)
            y.append(index)

    # add preprocessed images and labels to corresponding numpy arrays
    X = np.array(X)
    y = np.array(y)
    return X, y

def split_data(X, y):
    # split data 80/20 (for train and test sets)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # creat validation data by further splitting training data (80/20)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# simple random forest model
def train_random_forest(X_train, X_test, y_train, y_test):
    rf_classifier = RandomForestClassifier(n_estimators=10, random_state=42)
    rf_classifier.fit(X_train.reshape(len(X_train), -1), y_train)
    
    train_accuracy = rf_classifier.score(X_train.reshape(len(X_train), -1), y_train)
    test_accuracy = rf_classifier.score(X_test.reshape(len(X_test), -1), y_test)
    
    return rf_classifier, train_accuracy, test_accuracy

# create CNN model using transfer learning (VGG16)
def create_cnn_model(input_shape, num_classes):
    # load VGG16 without the top layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # freeze base layers
    for layer in base_model.layers:
        layer.trainable = False

    # add layers for classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.7)(x)  
    x = BatchNormalization()(x)  
    x = Dense(256, activation='relu')(x)  
    x = Dropout(0.5)(x)  
    x = BatchNormalization()(x)  
    x = Dense(128, activation='relu')(x) 
    x = Dropout(0.5)(x) 
    x = Dense(64, activation='relu')(x) 
    x = Dropout(0.5)(x) 
    x = BatchNormalization()(x) 
    predictions = Dense(num_classes, activation='softmax')(x)

    # create model given the specificed base model inputs and predictions
    model = Model(inputs=base_model.input, outputs=predictions)

    # define learning rate for better convergence
    learning_rate = 0.001

    # compile
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# main
if __name__ == "__main__":
    folder_path = "images"  
    X, y = preprocess_data(folder_path)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

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

    # create/compile CNN model
    cnn_model = create_cnn_model((224, 224, 3), 5)

    # use early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # train CNN model with early stopping
    history = cnn_model.fit(X_train, y_train, batch_size=32, epochs=200, validation_data=(X_val, y_val), callbacks=[early_stopping])

    # evaluate the model on train, validation, and test sets
    train_loss, train_accuracy = cnn_model.evaluate(X_train, y_train)
    val_loss, val_accuracy = cnn_model.evaluate(X_val, y_val)
    test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test)

    print("CNN Train Accuracy:", train_accuracy)
    print("CNN Validation Accuracy:", val_accuracy)
    print("CNN Test Accuracy:", test_accuracy)

    # access training history
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(train_accuracy) + 1)

    # Clear the current figure
    plt.clf()

    # Plot train and validation accuracy over epochs
    plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("train_validation_history.png")
    plt.show()

    # Save the model
    cnn_model.save("cnn_model.h5")
