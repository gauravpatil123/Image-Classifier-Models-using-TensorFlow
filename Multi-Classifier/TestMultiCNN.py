"""
TestMultiCNN:
    1. Loads trained model from the saved file
    2. Initializes and shuffles the test dataset list
    3. Predicts the test dataset using the trained model
    4. Calculates accuracy of the test dataset
"""

import tensorflow as tf
import os
import numpy as np
import DatasetDirectoryPreprocessing as DDP
import MultiCNN as CNN
from keras_preprocessing import image
import random

# loading trained model
Model = CNN.MultiCNN()
model = Model.load_model("multiCNN.h5")

# preprocessing test dataset
NUM_CLASSES = DDP.num_classes
CLASS_DICT = DDP.class_dict
TEST_DIR = DDP.test_dir
test_image_list = os.listdir(TEST_DIR)
try:
    test_image_list.remove(".DS_Store")
except:
    pass
num_test_images = len(test_image_list)
print("Number of Test Images: ", num_test_images)
random.shuffle(test_image_list)
print(test_image_list)

TOTAL_TEST_IMAGES = 0
accurate_images = 0

# predictions
for fn in test_image_list:
    path = os.path.join(TEST_DIR, fn)
    img = image.load_img(path, target_size = (300, 300))

    xs = image.img_to_array(img)
    xs = np.expand_dims(xs, axis = 0)

    #images = np.vstack([xs])
    classes = model.predict(xs)
    print(classes)

    for idx in range(NUM_CLASSES):
        if classes[0][idx] > 0.5:
            key = "n" + str(idx)
            print("\n" + fn + " is a " + CLASS_DICT.get(key))
            TOTAL_TEST_IMAGES += 1
            fn_label = fn[:2]
            if key == fn_label:
                accurate_images += 1

print("Total tested images = ", TOTAL_TEST_IMAGES)
accuracy = accurate_images / TOTAL_TEST_IMAGES
accuracy = accuracy * 100
print("Accuracy = " + str(accuracy) + "%")

