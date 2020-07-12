#!/usr/bin/python3.6

import cv2
import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import re

CATEGORIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

print(CATEGORIES)

path = "./data/test"
images = list(sorted(os.listdir(path)))     # Load test images


def preprocess(image):
    IMG_SIZE = 50
    img_array = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img_array = img_array / 255.0
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def predict(test_list):
    """
    Function takes a list of images to be tested and prints results
    Function returns a list of predicted values
    """
    pred_res = []
    print("Testing: ")
    model = tf.keras.models.load_model('./cnn.model')
    for img in images:
        test_img = os.path.join(path, img)
        preprocess(test_img)
        prediction = model.predict([preprocess(test_img)])
        res = prediction[0]
        max_nun = -1
        index = -1
        result_index = np.argmax(res)
        predict = CATEGORIES[result_index]
        print("Image: ", img)
        print("Predict: ", predict)
        pred_res.append(predict)
        print("\n")
    print(pred_res)
    return pred_res

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def getTrueLabels(test_list):
    """
    The function uses regex extracts label name out of image name
    Will only work if the label name is the only capital letter in 
    the image name. . .
    """
    true_labels = []
    for img in test_list:
        regex = re.findall("[A-Z]", img) # Extract capital word
        for match in regex:
            true_labels.append(match)
    return true_labels

# True labels of the test images
y_true = getTrueLabels(images)

# Predicted labels
y_pred = predict(images)


confusion_matrix(y_true, y_pred, labels=CATEGORIES)
print(confusion_matrix)

#plot_confusion_matrix(confusion_matrix, classes = CATEGORIES, title="Confussion matrix")
