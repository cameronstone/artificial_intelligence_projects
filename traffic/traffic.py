import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

# my pathname /Users/cameron/Desktop/artificial_intelligence_projects/gtsrb

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    
    # create lists to hold data
    images = []
    labels = []

    # make sure it is platform independent
    data_dir = data_dir.replace('/', os.sep)
    # IS THIS CORRECT FOR WINDOWS? I don't have one.
    data_dir = data_dir.replace('\\', os.sep)

    # get all 43 category directories
    for category_directory in range(0, NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category_directory))
        # get all files within each directory (category)
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            # get each image within each category
            image = cv2.imread(file_path)
            # resize image
            resized_image = cv2.resize(image, dsize=(IMG_WIDTH, IMG_HEIGHT))
            # add proper image and label to lists
            images.append(resized_image)
            labels.append(category_directory)

    # return tuple of resized images and their labels
    return (images, labels)

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
