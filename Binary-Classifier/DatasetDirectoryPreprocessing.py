"""
DatasetDirectoryPreprocessing:
    1. Configures the file paths and directories of the train, validation and test datasets
    2. prints out the length of the classes and combined length of classes in
       the train and validation datasets
"""
import os

# TODO: use standard logger instead of print, change the data structure to class
# TODO: after changing data structure, initialize the train, val, and test directories to be used globally

train_dir = "data/cats_and_dogs/training/" # critical to initialize after class defination
validation_dir = "data/cats_and_dogs/validation/" # critical to initialize after class defination
test_dir = "data/cats_and_dogs/test/mixed/" # critical to initialize after class defination

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

class DatasetDirectories:
    
    def __init__(self, train_dir, val_dir, test_dir):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.train_cats_dir = os.path.join(train_dir, "cats")
        self.train_dogs_dir = os.path.join(train_dir, "dogs")
        self.train_cats_fname = os.listdir(train_cats_dir)
        self.train_dogs_fname = os.listdir(train_dogs_dir)
        self.validation_cats_dir = os.path.join(validation_dir, "cats")
        self.validation_dogs_dir = os.path.join(validation_dir, "dogs")
        self.validation_cats_fname = os.listdir(validation_cats_dir)
        self.validation_dogs_fname = os.listdir(validation_dogs_dir)

    def __call__(self):
        # use standard logger here to print info

