"""
BinaryImageGenerators:
    1. Defines Generators class
    2. Initializes and calls a generator object of class Generators
    3. Initializes the train and validation generators using 
       the dataset directories and generator object, to be used globally
"""

import DatasetDirectoryPreprocessing as DDP
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAINING_DIR = DDP.train_dir
VALIDATION_DIR = DDP.validation_dir

class Generators:

    """
    class to create training and validation generators
    """

    def __init__(self):
        """
        Initializes the train and validation datagens to create generators
        """
        self.train_datagen = ImageDataGenerator(rescale=1./255)
        self.validation_datagen = ImageDataGenerator(rescale=1./255)

    def create_train_generator(self, dataset_dir, target_size, batch_size):
        """
        Input:
            dataset_dir: directory path for the training dataset
            target_size: target size of the dataset images
            batch_size: batch size for flowing images through the generator

        Output:
            Returns the train_generator which flows images from the directory
        """
        train_generator = self.train_datagen.flow_from_directory(dataset_dir, 
                                                                 target_size = target_size,
                                                                 batch_size = batch_size,
                                                                 class_mode = 'binary'
                                                                 )
        return train_generator
    
    def create_val_generator(self, dataset_dir, target_size, batch_size):
        """
        Input:
            dataset_dir: directory path for the validation dataset
            target_size: target size of the dataset images
            batch_size: batch size for flowing images through the generator
        
        Output:
            Returns the validatin_generator which flows images from the directory
        """
        validation_generator = self.validation_datagen.flow_from_directory(dataset_dir,
                                                                           target_size = target_size,
                                                                           batch_size = batch_size,
                                                                           class_mode = 'binary'
                                                                           )
        return validation_generator

# initializing generators to be used in training
generators = Generators()
train_generator = generators.create_train_generator(TRAINING_DIR, (150, 150), 200)
validation_generator = generators.create_val_generator(VALIDATION_DIR, (150, 150), 200)
