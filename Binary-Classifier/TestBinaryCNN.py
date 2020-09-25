"""
TestBinaryCNN:
    1. Loads the trained model
    2. Loads the test directory and test list
    3. logs info about model and test set
    4. classifies the images in test set and logs their classes
"""

import os
import tensorflow as tf
import matplotlib.image as mpimg
from keras_preprocessing import image
import BinaryCNN as CNN
import DatasetDirectoryPreprocessing as DDP
import numpy as np
import logging

Model = CNN.BinaryCNN()
model = Model.load_model("binaryCNN.h5")

# testing model on test set
TEST_DIR = DDP.test_dir
TEST_LIST  = os.listdir(TEST_DIR)

logging.basicConfig(format='%(message)s', level=logging.INFO)
message = "\nEvaluating model on test set"
test_set_log = "\nTest set size = " + str(len(TEST_LIST))
logging.info(message)
logging.info(test_set_log)

for fn in TEST_LIST:
    # predicting images
    path = os.path.join(TEST_DIR + fn)
    img = image.load_img(path, target_size=(150, 150))

    xs = image.img_to_array(img)
    xs = np.expand_dims(xs, axis = 0)

    images = np.vstack([xs])
    classes = model.predict(images, batch_size = 20)
    
    #print(fn) #TODO: Uncomment to print Image name
    #print(classes) #TODO: Uncomment to print classes
    
    if classes[0] < 0.5:
        class_log = "\n" + fn + " is a cat"
        logging.info(class_log)
    else:
        class_log = "\n" + fn + " is a dog"
        logging.info(class_log)
