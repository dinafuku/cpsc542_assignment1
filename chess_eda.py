import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# check versions of each librar
import sklearn
print(sklearn.__version__)
import tensorflow as tf
print(tf.__version__)
from tensorflow import keras as kb
print(kb.__version__)

# define the path to image data and eda images
data_dir = "images"
eda_images_dir = "eda_images"

# load image data
def load_image_data(data_dir):
    X = []
    y = []

    # folder labels
    labels = {'pawn': 0, 'bishop': 1, 'knight': 2, 'rook': 3, 'queen': 4}

    # iterate through each folder and each image and add to numpy arrays
    for label, index in labels.items():
        images = os.listdir(os.path.join(data_dir, label))
        for image_name in images:
            image_path = os.path.join(data_dir, label, image_name)
            image = Image.open(image_path)
            X.append(np.array(image))
            y.append(index)

    # convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)
    return X, y

# eda analysis
def perform_eda(X, y):
    # create directory for sample images
    os.makedirs(eda_images_dir, exist_ok=True)

    # display the distribution of samples in each class with a bar graph
    class_labels = ['pawn', 'bishop', 'knight', 'rook', 'queen']
    class_counts = np.bincount(y)
    plt.figure(figsize=(10, 6))
    custom_palette = sns.color_palette("pastel", n_colors=len(class_labels))
    sns.barplot(x=class_labels, y=class_counts, palette=custom_palette)
    plt.title("Number of Samples in Each Class")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.savefig(os.path.join(eda_images_dir, "class_samples.png"))  
    plt.clf()  

    # display sample images from each class
    for i, label in enumerate(class_labels):
        # create subdirectory for each class label
        class_dir = os.path.join(eda_images_dir, label)
        os.makedirs(class_dir, exist_ok=True)
        
        # save 10 sample images from each class
        for j, image in enumerate(X[y == i][:10]):
            plt.imshow(image)
            plt.title(f"{label}_{j+1}")
            plt.axis("off")
            plt.savefig(os.path.join(class_dir, f"{label}_{j+1}.png"))  
            plt.clf()  

if __name__ == "__main__":
    # load the image data
    X, y = load_image_data(data_dir)

    # conduct exploratory data analysis
    perform_eda(X, y)
