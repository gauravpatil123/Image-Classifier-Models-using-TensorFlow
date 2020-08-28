"""
DatasetDirectoryPreprocessing:
    1. Defines DatasetDirectory class to initialize directories and datasets
    2. Initializes a directories, dictionary 'class_dict' to store all the classess and logs number of classes 
"""
import os
import logging

class DatasetDirectories:

    """
    class to initialize the dataset directories
    """

    def __init__(self, base_dir, test_dir):
        """
        Input:
            base_dir: base directory path for train and validation datasets
            test_dir: directory path for test dataset

        Action:
            Initializes the directories and dataset in class
        """
        self.base_dir = base_dir
        self.test_dir = test_dir
        self.class_dict = { "n0" : "mantled howler",
                            "n1" : "patas monkey",
                            "n2" : "bald uakari",
                            "n3" : "japanese macaque",
                            "n4" : "pygmy marmoset",
                            "n5" : "white headed capuchin",
                            "n6" : "silvery marmoset",
                            "n7" : "common squirrel monkey",
                            "n8" : "black headed night monkey",
                            "n9" : "nilgiri langur"}
        self.train_dir = os.path.join(self.base_dir, "training/")
        self.validation_dir = os.path.join(self.base_dir, "validation/")
        self.num_classes = len(self.class_dict)

    def __call__(self):
        """
        logs number of classes
        """
        class_dict = self.class_dict
        num_classes = len(class_dict)
        log_classes = "Number of classes: " + str(num_classes)
        logging.basicConfig(format="%(message)s", level=logging.INFO)
        logging.info(log_classes)

    def get_directories(self):
        """
        Returns train, validation and test directories
        """
        return self.train_dir, self.validation_dir, self.test_dir

    def get_classes(self):
        """
        Returns a dictionary of class labels
        """
        return self.class_dict

    def get_num_classes(self):
        """
        Returns the number of classes
        """
        return self.num_classes

dataset_dir = DatasetDirectories("data/Monkey-Species/", "data/testing/")
train_dir, validation_dir, test_dir = dataset_dir.get_directories()
num_classes = dataset_dir.get_num_classes()
class_dict = dataset_dir.get_classes()
dataset_dir()
    
"""
base_dir = "data/Monkey-Species/"

train_dir = os.path.join(base_dir, "training/")
validation_dir = os.path.join(base_dir, "validation/")
test_dir = "data/testing/"

class_dict = {"n0" : "mantled howler",
              "n1" : "patas monkey",
              "n2" : "bald uakari",
              "n3" : "japanese macaque",
              "n4" : "pygmy marmoset",
              "n5" : "white headed capuchin",
              "n6" : "silvery marmoset",
              "n7" : "common squirrel monkey",
              "n8" : "black headed night monkey",
              "n9" : "nilgiri langur"}

num_classes = len(class_dict)
print("Number of classes: ", num_classes)
"""
