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
logging.info(str(test_image_list))

TOTAL_TEST_IMAGES = 0
accurate_images = 0

for fn in test_image_list:
    path = os.path.join(TEST_DIR, fn)
    img = image.load_img(path, target_size = (300, 300))

    xs = image.img_to_array(img)
    xs = np.expand_dims(xs, axis = 0)

    classes = model.predict(xs)
    logging.info(str(classes))

    for idx in range(NUM_CLASSES):
        if classes[0][idx] > 0.5:
            key = "n" + str(idx)
            message = "\n" + str(fn) + " is a " + str(CLASS_DICT.get(key))
            logging.info(message)
            TOTAL_TEST_IMAGES += 1
            fn_label = fn[:2]
            if key == fn_label:
                accurate_images += 1


test_images_log = "Total tested images = " + str(TOTAL_TEST_IMAGES)
logging.info(test_images_log)
accuracy = accurate_images / TOTAL_TEST_IMAGES
accuracy = accuracy * 100
acc_log = "Accuracy = " + str(accuracy) + "%"
logging.info(acc_log)


