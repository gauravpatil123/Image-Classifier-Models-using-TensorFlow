from tensorflow.keras.preprocessing.image import ImageDataGenerator
import DatasetDirectoryPreprocessing as DDP

TRAIN_DIR = DDP.train_dir
VALIDATION_DIR = DDP.validation_dir

train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   fill_mode = 'nearest')

validation_datagen = ImageDataGenerator(rescale = 1./255.)

# Flowing Images in generator using datagens
train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                    batch_size = 100,
                                                    class_mode = 'categorical',
                                                    target_size = (150, 150))

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size = 50,
                                                              class_mode = 'categorical',
                                                              target_size = (150, 150))
