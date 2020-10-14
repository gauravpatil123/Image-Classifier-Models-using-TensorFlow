"""
TestMultiCNN:
    1. Loads trained model from the saved file
    2. Initializes and shuffles the test dataset list
    3. predicts  the test dataset using the trained model
    4. calculates accuracy of the test dataset
"""

import tensorflow as tf
import os
import numpy as np
import DatasetDirectoryPreprocessing as DDP
from keras_preprocessing import image
import random
import logging

# standard logger configuration
logging.basicConfig(format="%(message)s", level=logging.INFO)

# loading trained model
model = tf.keras.models.load_model("multiCNN.h5")

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
test_images_log = "Number of Test Images: " + str(num_test_images)
logging.info(test_images_log) 
random.shuffle(test_image_list)

def predict(verbose=False):
    """
    Predicts the classes on test set using the trained model

    Input:
        verbose: boolean to print the result for each image
    """
    TOTAL_TEST_IMAGES = 0
    accurate_images = 0

    # predictions
    for fn in test_image_list:
        path = os.path.join(TEST_DIR, fn)
        img = image.load_img(path, target_size = (300, 300))

        xs = image.img_to_array(img)
        xs = np.expand_dims(xs, axis = 0)

        classes = model.predict(xs)

        for idx in range(NUM_CLASSES):
            if classes[0][idx] > 0.5:
                key = "n" + str(idx)
                if verbose:
                    class_name = str(CLASS_DICT.get(key))
                    message = "\n" + fn + " is a " + class_name
                    logging.info(message)
                TOTAL_TEST_IMAGES += 1
                fn_label = fn[:2]
                if key == fn_label:
                    accurate_images += 1

    total_tested_img_log = "Total tested images = " + str(TOTAL_TEST_IMAGES)
    logging.info(total_tested_img_log)
    accuracy = accurate_images / TOTAL_TEST_IMAGES
    accuracy = accuracy * 100
    accuracy_log = "Accuracy = " + str(accuracy) + "%"
    logging.info(accuracy_log)

predict()
# predict(verbose=True)
