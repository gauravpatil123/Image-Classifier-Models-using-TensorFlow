"""
ImageGenerators:
   1. Defines Generators class to initialize ImageDataGenerators for the 
      training and validation datagens, Using Data-Augmentation parameters to 
      add synthetic data in the training dataset
   2. Initializes the train and validation generators to be used in training the model
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import DatasetDirectoryPreprocessing as DDP

TRAIN_DIR = DDP.train_dir
VALIDATION_DIR = DDP.validation_dir

class Generators:

   def __init__(self, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, 
               shear_range=0.2, zoom_range=0.2, horizontal_flip = True, fill_mode='nearest'):
      """
      Input:
         rotation_range: proportion range of rotaion of images
         width_shift_range: proportion of width shift for images
         height_shift_range: proportion of height shift for images
         shear_range: proportion of shear for images
         zoom_range: proportion of zoom for images
         horizontal_flip: boolean to set horizontal flipped images
         fill_mode: type of fill of image voids

      Action:
         1. Initializes the train datagen with data augmentation
         2. Initializes the validation datagen
      """
      self.train_datagen = ImageDataGenerator(rescale = 1./255,
                                              rotation_range = rotation_range,
                                              width_shift_range = width_shift_range,
                                              height_shift_range = height_shift_range,
                                              shear_range = shear_range,
                                              zoom_range = zoom_range,
                                              horizontal_flip = horizontal_flip,
                                              fill_mode = fill_mode)
      
      self.validation_datagen = ImageDataGenerator(rescale = 1./255)

   def create_train_generator(self, dataset_dir, batch_size , target_size, class_mode = 'categorical'):
      """
      Input:
         dataset_dir: directory path for the training dataset
         target_size: target size for the images
         batch_size: batch size for training dataset
         class_mode: class mode for training
      
      Output:
         train_generator: train generator for for flowing train images to model
      """
      train_generator = self.train_datagen.flow_from_directory(dataset_dir,
                                                               batch_size = batch_size,
                                                               class_mode = class_mode,
                                                               target_size = target_size)
      return train_generator

   def create_val_generator(self, dataset_dir, batch_size, target_size, class_mode = 'categorical'):
      """
      Input:
         dataset_dir: directory path for the training dataset
         target_size: target size for the images
         batch_size: batch size for training dataset
         class_mode: class mode for training
      
      Output:
         validation_generator: validation generator for for flowing train images to model
      """
      validation_generator = self.validation_datagen.flow_from_directory(dataset_dir,
                                                                         batch_size = batch_size,
                                                                         class_mode = class_mode,
                                                                         target_size = target_size)
      return validation_generator

# initializing generators for training
datagens = Generators()
train_generator = datagens.create_train_generator(TRAIN_DIR, 50, (300, 300), class_mode='categorical')
validation_generator = datagens.create_val_generator(VALIDATION_DIR, 25, (300, 300), class_mode='categorical')
