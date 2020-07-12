#/!usr/bin/python3.6

import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import random

path = "./data/train"     # Data dir
limit = 6           # Number of gestures to be loaded
IMG_SIZE = 50       # Shape of processed image
labels = [x for x in range(limit)]

# Function returns a list with path of all images to 
def create_training_data(path, limit):
    print("creating training data")
    train_data = []
    # List of all gestures in train folder
    gestures = list(sorted(os.listdir(path)))  
    count = 0
    for ges in gestures:
       # if count >= limit:
       #     print("Train data created . . .")
       #     print("Total Images: {}".format(len(train_data)))
       #     return train_data
        print("Current Gesture: {}, label: {}".format(ges, count))
        # Path of geture dir
        ges_path = os.path.join(path, ges)
        for img in os.listdir(ges_path):
            img_path = os.path.join(ges_path, img)
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # resize
            img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            train_data.append([img_array, count])

        count += 1
    return train_data

# Save the training data to .npy file
def save_data(train_data):
    print("saving data")
    random.shuffle(train_data)    # shuffle training data
    X = []
    y = []
    for features, label in train_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    np.save('features.npy', X)
    np.save('labels.npy', y)
    print("Features and labels saved on disk . . . ")


# List contains numpy array of all the training images
train_data = create_training_data(path, limit)
save_data(train_data)
