import tensorflow as tf
import os
import numpy as np
import DatasetDirectoryPreprocessing as DDP
from keras_preprocessing import image

# loading trained model
model = tf.keras.models.load_model("multiCNN.h5")

# preprocessing test dataset
NUM_CLASSES = DDP.num_classes
CLASS_DICT = DDP.class_dict
TEST_DIR = DDP.test_dir
test_image_list = os.listdir(TEST_DIR)
test_image_list.remove(".DS_Store")
num_test_images = len(test_image_list)
print("Number of Test Images: ", num_test_images)
print(test_image_list)

# predictions
for fn in test_image_list:
    path = os.path.join(TEST_DIR, fn)
    img = image.load_img(path, target_size = (300, 300))

    xs = image.img_to_array(img)
    xs = np.expand_dims(xs, axis = 0)

    #images = np.vstack([xs])
    classes = model.predict(xs)
    #print(classes)

    for idx in range(NUM_CLASSES):
        if classes[0][idx] > 0.5:
            key = "n" + str(idx)
            print("\n" + fn + " is a " + CLASS_DICT.get(key))

