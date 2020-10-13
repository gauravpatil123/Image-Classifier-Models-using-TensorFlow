"""
DatasetDirectoryPreprocessing:
    1. Prints out the length of the classes and combined length of classes in
       the train and validation datasets using standard logger
    2. Initailizes and calls a log_directory object of class DatasetDirectories
    3. Initializes train, validation and test directory paths to be used globally
"""
import os
import logging

class DatasetDirectories:
    
    """
    class to log and setup the filepaths for the training, validation and test datasets
    """

    def __init__(self, train_dir, val_dir, test_dir):
        """
        Input:
            train_dir: directory path dor training dataset
            val_dir: directory path for validation dataset
            test_dir: directory path for test dataset

        Action:
            Initializes the train and validation directories 
            along with both the classification classes
        """
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.train_cats_dir = os.path.join(train_dir, "cats")
        self.train_dogs_dir = os.path.join(train_dir, "dogs")
        self.train_cats_fname = os.listdir(self.train_cats_dir)
        self.train_dogs_fname = os.listdir(self.train_dogs_dir)
        self.validation_cats_dir = os.path.join(val_dir, "cats")
        self.validation_dogs_dir = os.path.join(val_dir, "dogs")
        self.validation_cats_fname = os.listdir(self.validation_cats_dir)
        self.validation_dogs_fname = os.listdir(self.validation_dogs_dir)

    def __call__(self):
        """
        prints the training and validation dataset statistics in command prompt
        """
        logging.basicConfig(format='%(message)s', level=logging.INFO)
        train_cats_log = "\nTraining Cats Images = " + str(len(self.train_cats_fname))
        train_dogs_log = "\nTraining Dogs Images = " + str(len(self.train_dogs_fname))
        train_log = "\nTotal training Images = " + str(len(self.train_cats_fname) + len(self.train_dogs_fname))
        logging.info(train_cats_log)
        logging.info(train_dogs_log)
        logging.info(train_log)
        val_cats_log = "\nValidation Cats Images = " + str(len(self.validation_cats_fname))
        val_dogs_log = "\nValidation Dogs Images = " + str(len(self.validation_dogs_fname))
        val_log = "\nTotal validation Images = " + str(len(self.validation_cats_fname) + len(self.validation_dogs_fname))
        logging.info(val_cats_log)
        logging.info(val_dogs_log)
        logging.info(val_log)

    def get_train_dir(self):
        """
        Returns the directory path for training dataset
        """
        return self.train_dir

    def get_val_dir(self):
        """
        Returns the directory path for validation dataset
        """
        return self.val_dir

    def get_test_dir(self):
        """
        Returns the directory path for test dataset
        """
        return self.test_dir

    def get_test_cat_dir(self):
        """
        Returns the directory path for cat images in test directory
        """
        cat_dir = self.test_dir + "cats/"
        return cat_dir

    def get_test_dog_dir(self):
        """
        Returns the directory path for dog images in test directory
        """
        dog_dir = self.test_dir + "dogs/"
        return dog_dir

# Initializing directories to be used in training
log_directories = DatasetDirectories("data/cats_and_dogs/training/", "data/cats_and_dogs/validation/", "data/cats_and_dogs/test/mixed/")
log_directories()

train_dir = log_directories.get_train_dir()
validation_dir = log_directories.get_val_dir()
test_dir = log_directories.get_test_dir()
test_cat_dir = log_directories.get_test_cat_dir()
test_dog_dir = log_directories.get_test_dog_dir()
