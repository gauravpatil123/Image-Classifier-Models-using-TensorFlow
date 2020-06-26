import tensorflow as tf
import os
import numpy as np
#import DatasetDirectoryPreprocessing as DDP
from keras_preprocessing import image

# loading trained model
model = tf.keras.models.load_model("multiCNN.h5")

# test datatset configuration
TEST_DIR = "data/testing/"
CLASS_LIST = os.listdir(TEST_DIR)
CLASS_LIST = sorted(CLASS_LIST)

try:
    CLASS_LIST.remove(".DS_Store")
except:
    pass

NUM_CLASSES = 150
FAILED_IMAGES = 0
TOTAL_TEST_IMAGES = 0
accurate_images = 0

for classname in CLASS_LIST:
    print("\nPrediction for images in " + classname + " class")
    classpath = os.path.join(TEST_DIR, classname)
    classpath_list = os.listdir(classpath)

    for fn in classpath_list:
        img_path = os.path.join(classpath, fn)
        try:
            img = image.load_img(img_path, target_size = (300, 300))

            xs = image.img_to_array(img)
            xs = np.expand_dims(xs, axis = 0)

            classes = model.predict(xs)
            #print("\n")
            #print(classes)

            for idx in range(NUM_CLASSES):
                if classes[0][idx] > 0.5:
                    key = CLASS_LIST[idx]
                    #print("class index = " , idx)
                    print("\n" + fn + " is a " + key)
                    TOTAL_TEST_IMAGES += 1
                    if key == classname:
                        accurate_images += 1
        except:
            print("Failed to load " + fn + " image")
            FAILED_IMAGES += 1
        
print("Total tested images = ", TOTAL_TEST_IMAGES)
print("Failed images = ", FAILED_IMAGES)
accuracy = accurate_images / TOTAL_TEST_IMAGES
accuracy = accuracy * 100
print("Accuracy = " + str(accuracy) + "%")


"""
TEST_DIR = DDP.test_dir
CLASS_LIST = DDP.class_list
TEST_IMAGE_COUNT = 0

for folder in CLASS_LIST:
    # predictions
    print("Predictions for images in " + folder + " class")
    path = os.path.join(TEST_DIR, folder)
    path_list = os.listdir(path)

    for fn in path_list:
        img_path = os.path.join(path, fn)
        try:
            img = image.load_img(img_path, target_size = (300, 300))

            xs = image.img_to_array(img)
            xs = np.expand_dims(xs, axis = 0)

            images = np.vstack([xs])
            classes = model.predict(images, batch_size = 20)
            # print(classes)

            TEST_IMAGE_COUNT += 1
            CLASS_IMAGE_INDEX = 0
            print("Test Image no.: ", TEST_IMAGE_COUNT)
            for idx in range(150):
                if classes[CLASS_IMAGE_INDEX][idx] > 0.6:
                    print("\n" + fn + " is a " + CLASS_LIST[idx])
        
            CLASS_IMAGE_INDEX += 1
        
        except:
            continue
"""


