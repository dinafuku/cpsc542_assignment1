# import needed libraries
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
    # datagenerator that conducts data augmentation and preprocessing steps for the images
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=542)
    
    # create validation data by further splitting training data (80/20)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=542)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# simple random forest model
def train_random_forest(X_train, X_test, y_train, y_test):
    # define random forest model with a low amount of decision trees due to signifcant overfitting
    rf_classifier = RandomForestClassifier(n_estimators=10, random_state=542)
    # reshape training data since we need to convert image shape and flatten
    rf_classifier.fit(X_train.reshape(len(X_train), -1), y_train)
    
    # output training and test accuracy of the model
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
    # dropout for regulurization
    x = Dropout(0.7)(x)  
    x = Dense(256, activation='relu')(x)  
    x = Dropout(0.5)(x)  
    x = Dense(128, activation='relu')(x) 
    x = Dropout(0.5)(x) 
    x = Dense(64, activation='relu')(x) 
    x = Dropout(0.5)(x) 
    # batch norm also for normalization
    x = BatchNormalization()(x) 
    # use soft max since we are predicted between 5 classes not just 2
    predictions = Dense(num_classes, activation='softmax')(x)

    # create model given the specificed base model inputs and predictions
    model = Model(inputs=base_model.input, outputs=predictions)

    # define learning rate for better convergence (prevent large skips over minima)
    learning_rate = 0.001

    # compile using specified learning rate and track accuracy
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# main
if __name__ == "__main__":
    # provide data folder path
    folder_path = "images"  
    X, y = preprocess_data(folder_path)

    # split data into train, test, validation
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # train simple rf model
    rf_model, train_accuracy, test_accuracy = train_random_forest(X_train, X_test, y_train, y_test)

    # reshape training data for rf model
    rf_y_pred = rf_model.predict(X_test.reshape(len(X_test), -1))
    # compute confusion matrix for rf model
    conf_matrix = confusion_matrix(y_test, rf_y_pred)

    # print out confusion matrix for rf model
    print("Random Forest Train Accuracy:", train_accuracy)
    print("Random Forest Test Accuracy:", test_accuracy)
    print(conf_matrix)

    # plot confusion matrix using seaborn for easier visability
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

    # print out those metrics
    print("CNN Train Accuracy:", train_accuracy)
    print("CNN Validation Accuracy:", val_accuracy)
    print("CNN Test Accuracy:", test_accuracy)

    # access training history
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(train_accuracy) + 1)

    # clear the current figure
    plt.clf()

    # plot train and validation accuracy over epochs
    plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("train_validation_history.png")
    plt.show()

    # clear the current figure
    plt.clf()

    # save the model
    cnn_model.save("cnn_model.h5")

    # load saved model
    loaded_model = load_model("cnn_model.h5")

    # predict labels for the test set using the loaded CNN model
    cnn_y_pred = loaded_model.predict(X_test)
    cnn_y_pred_classes = np.argmax(cnn_y_pred, axis=1)

    # generate confusion matrix for CNN
    cnn_conf_matrix = confusion_matrix(y_test, cnn_y_pred_classes)

    # plot confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cnn_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix - CNN")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix_cnn.png") 
    plt.show()

    plt.clf()

    # create directory for the predicted images
    predicted_images_dir = "predicted_images"
    os.makedirs(predicted_images_dir, exist_ok=True)

    # predict image class labels with cnn
    for i in range(10):
        # select random index from test set
        index = np.random.randint(0, len(X_test))

        # get corresponding image and label
        image = X_test[index]
        true_label = y_test[index]

        # predict label using CNN
        predicted_label = np.argmax(loaded_model.predict(np.expand_dims(image, axis=0)))

        # save images
        plt.imshow(image)
        plt.title(f"True Label: {labels[true_label]}, Predicted Label: {labels[predicted_label]}")
        plt.axis('off')  # Remove tick marks and numbers
        plt.savefig(os.path.join(predicted_images_dir, f"predicted_image_{i}.png"))
        plt.show()
        plt.clf()