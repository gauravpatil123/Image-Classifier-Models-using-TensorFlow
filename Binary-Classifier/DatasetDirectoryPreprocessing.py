"""
DatasetDirectoryPreprocessing:
    1. Configures the file paths and directories of the train, validation and test datasets
    2. prints out the length of the classes and combined length of classes in
       the train and validation datasets
"""
import os
# import zipfile

train_dir = "data/cats_and_dogs/training/"
validation_dir = "data/cats_and_dogs/validation/"
test_dir = "data/cats_and_dogs/test/mixed/"

train_cats_dir = os.path.join(train_dir, "cats")
train_dogs_dir = os.path.join(train_dir, "dogs")
train_cats_fname = os.listdir(train_cats_dir)
train_dogs_fname = os.listdir(train_dogs_dir)
print("\nTraining Cats Images = ", len(train_cats_fname))
print("\nTraining Dogs Images = ", len(train_dogs_fname))
print("\nTotal training Images =", len(train_cats_fname) + len(train_dogs_fname))

validation_cats_dir = os.path.join(validation_dir, "cats")
validation_dogs_dir = os.path.join(validation_dir, "dogs")
validation_cats_fname = os.listdir(validation_cats_dir)
validation_dogs_fname = os.listdir(validation_dogs_dir)
print("\nValidation Cats Images = ", len(validation_cats_fname))
print("\nValidation Dogs Images = ", len(validation_dogs_fname))
print("\nTotal validation Images = ", len(validation_cats_fname) + len(validation_dogs_fname))


