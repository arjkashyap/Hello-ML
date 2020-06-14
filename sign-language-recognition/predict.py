#!/usr/bin/python3.6

import cv2
import tensorflow as tf
import os


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



# Function takes a list of images to be tested and prints result
def predict(test_list):
    print("Testing: ")
    model = tf.keras.models.load_model('./cnn.model')
    for img in images:
        test_img = os.path.join(path, img)
        preprocess(test_img)
        prediction = model.predict([preprocess(test_img)])
        res = prediction[0]
        max_nun = -1
        index = -1
        for i in range(len(res)):
            if res[i] > max_nun:
                max_nun = res[i]
                index = i
        print("Image: ", img)
        print("Prediction: {}".format(CATEGORIES[index]))
        print("\n")

predict(images)
