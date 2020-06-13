import DatasetDirectoryPreprocessing as DDP
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAINING_DIR = DDP.train_dir
VALIDATION_DIR = DDP.validation_dir

# defining train_datagen and validation_datagen
train_datagen = ImageDataGenerator(rescale=1./255)

validation_datagen = ImageDataGenerator(rescale=1./255)

# defining train and validation generators from respective datagens to flow images
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    target_size = (150, 150),
                                                    batch_size = 20,
                                                    class_mode = 'binary'
                                                    )

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              target_size = (150, 150),
                                                              batch_size = 20,
                                                              class_mode = 'binary'
                                                              )

