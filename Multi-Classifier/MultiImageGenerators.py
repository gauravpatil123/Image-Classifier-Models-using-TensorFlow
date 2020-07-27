"""
MultiImageGenerators:
    1. Initializes ImageDataGenerators for the training and validation datagens,
       Using Data-Augmentation parameters to add synthetic data in the training dataset
    2. Uses the datagens to initialize train and validation generators to flow the Images
       to the model
"""

import DatasetDirectoryPreprocessing as DDP
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAINING_DIR = DDP.train_dir
VALIDATION_DIR = DDP.validation_dir

# defining train_datagen and validation_datagen with data augmentation
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   fill_mode = 'nearest')

validation_datagen = ImageDataGenerator(rescale = 1./255)

# defining train and validation generators form datagens to flow images
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    target_size = (300, 300),
                                                    batch_size = 50,
                                                    class_mode = 'categorical'
                                                    )

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              target_size = (300, 300),
                                                              batch_size = 50,
                                                              class_mode = 'categorical'
                                                              )
