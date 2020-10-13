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
TEST_LIST = os.listdir(TEST_DIR)
TEST_CAT_DIR = DDP.test_cat_dir
TEST_DOG_DIR = DDP.test_dog_dir
TEST_CAT_LIST = os.listdir(TEST_CAT_DIR)
TEST_DOG_LIST = os.listdir(TEST_DOG_DIR)

logging.basicConfig(format='%(message)s', level=logging.INFO)
message = "\nEvaluating model on test set"
test_set_log = "\nTest set size = " + str(len(TEST_CAT_DIR) + len(TEST_DOG_LIST))
logging.info(message)
logging.info(test_set_log)

if ".DS_Store" in TEST_LIST:
    TEST_LIST.remove(".DS_Store")

if ".DS_Store" in TEST_CAT_LIST:
    TEST_CAT_LIST.remove(".DS_Store")

if ".DS_Store" in TEST_DOG_LIST:
    TEST_DOG_LIST.remove(".DS_Store")

def test_cat_stats(verbose):
    """
    Runs the model on test set and calculates the stats on the model's performance as a Cat identifier

    Input:
        verbose: Boolean to switch on command prompt message for each image test
    """
    epsilon = 0.001
    true_positives = epsilon
    true_negatives = epsilon
    false_positives = epsilon
    false_negatives = epsilon

    for fn in TEST_CAT_LIST:
        # concatenating to file path
        path = os.path.join(TEST_CAT_DIR + fn)
        img = image.load_img(path, target_size=(150, 150))

        xs = image.img_to_array(img)
        xs = np.expand_dims(xs, axis = 0)

        images = np.vstack([xs])
        classes = model.predict(images, batch_size = 20)
    
        if classes[0] < 0.5:
            if verbose:
                class_log = "\n" + fn + " is a cat"
                logging.info(class_log)
            true_positives += 1 
        else:
            if verbose:
                class_log = "\n" + fn + " is a dog"
                logging.info(class_log)
            false_positives += 1 

    for fn in TEST_DOG_LIST:
        # concatenating to file path
        path = os.path.join(TEST_DOG_DIR + fn)
        img = image.load_img(path, target_size=(150, 150))

        xs = image.img_to_array(img)
        xs = np.expand_dims(xs, axis = 0)

        images = np.vstack([xs])
        classes = model.predict(images, batch_size = 20)
    
        if classes[0] < 0.5:
            if verbose:
                class_log = "\n" + fn + " is a cat"
                logging.info(class_log)
            false_negatives += 1 
        else:
            if verbose:
                class_log = "\n" + fn + " is a dog"
                logging.info(class_log)
            true_negatives += 1 

    # Calculating Precision and Recall
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    F1_Score = 2 * ((precision * recall) / (precision + recall))
    precision = '%.3f'%precision
    recall = '%.3f'%recall
    F1_Score = '%.3f'%F1_Score

    precision_log = "\nPrecision = " + str(precision)
    recall_log = "\nRecall = " + str(recall)
    f1_log = "\nF1 Score = " + str(F1_Score)
    message = "\nTest Set results on Model as a Cat Identifier"

    logging.info(message)
    logging.info(precision_log)
    logging.info(recall_log)
    logging.info(f1_log)

def test_doc_stats(verbose):
    """
    Runs the model on test set and calculates the stats on the model's performance as a Dog identifier
    """
    epsilon = 0.001
    true_positives = epsilon
    true_negatives = epsilon
    false_positives = epsilon
    false_negatives = epsilon

    for fn in TEST_CAT_LIST:
        # concatenating to file path
        path = os.path.join(TEST_CAT_DIR + fn)
        img = image.load_img(path, target_size=(150, 150))

        xs = image.img_to_array(img)
        xs = np.expand_dims(xs, axis = 0)

        images = np.vstack([xs])
        classes = model.predict(images, batch_size = 20)
    
        if classes[0] < 0.5:
            if verbose:
                class_log = "\n" + fn + " is a cat"
                logging.info(class_log)
            true_negatives += 1
        else:
            if verbose:
                class_log = "\n" + fn + " is a dog"
                logging.info(class_log)
            false_negatives += 1

    for fn in TEST_DOG_LIST:
        # concatenating to file path
        path = os.path.join(TEST_DOG_DIR + fn)
        img = image.load_img(path, target_size=(150, 150))

        xs = image.img_to_array(img)
        xs = np.expand_dims(xs, axis = 0)

        images = np.vstack([xs])
        classes = model.predict(images, batch_size = 20)
    
        if classes[0] < 0.5:
            if verbose:
                class_log = "\n" + fn + " is a cat"
                logging.info(class_log)
            false_positives += 1
        else:
            if verbose:
                class_log = "\n" + fn + " is a dog"
                logging.info(class_log)
            true_positives += 1

    # Calculating Precision and Recall
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    F1_Score = 2 * ((precision * recall) / (precision + recall))
    precision = '%.3f'%precision
    recall = '%.3f'%recall
    F1_Score = '%.3f'%F1_Score

    precision_log = "\nPrecision = " + str(precision)
    recall_log = "\nRecall = " + str(recall)
    f1_log = "\nF1 Score = " + str(F1_Score)
    message = "\nTest Set results on Model as a Dog Identifier"

    logging.info(message)
    logging.info(precision_log)
    logging.info(recall_log)
    logging.info(f1_log)

test_doc_stats(verbose=False)
test_cat_stats(verbose=False)

"""
TEST RESULTS:

DOG STATS

Precision = 0.940

Recall = 0.712

F1 Score = 0.810

CAT STATS
Precision = 0.620

Recall = 0.912

F1 Score = 0.738
"""
