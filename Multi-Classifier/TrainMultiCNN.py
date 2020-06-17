import tensorflow as tf
import MultiImageGenerators as MIG
import MultiCNN as CNN
import PlotCode as PC
import DatasetDirectoryPreprocessing as DDP
import numpy as np
import os
import matplotlib.pyplot as plt
from keras_preprocessing import image

# desired Training accuracy
DESIRED_TRAINING_ACC = 0.999

# configuring callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > DESIRED_TRAINING_ACC):
            print("\n Reached desired traininh accuracy of "+str(DESIRED_TRAINING_ACC * 100)+ "%, so cancelling further training")
            self.model.stop_training = True

callbacks = myCallback()

# CNN model
model = CNN.model

# model summary
model.summary()

# Image Generators
TRAIN_GENERATOR = MIG.train_generator
VALIDATION_GENERATOR = MIG.validation_generator

# compiling model
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

# training model
history = model.fit(TRAIN_GENERATOR,
                    validation_data = VALIDATION_GENERATOR,
                    steps_per_epoch = 577, # 577 x 7 = 4036
                    epochs = 200, 
                    validation_steps = 16, # 124 x 16 = 1984
                    verbose = 1,
                    callbacks = [callbacks])

# saving trained weights
model.save("multiCNN.h5")

# plotting results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

PC.plot(acc, epochs, 'accuracy', 'train', 'validation', 'g', 'b')
PC.plot(loss, val_loss, epochs, 'loss', 'train', 'validation', 'r', 'orange')

# testing model on test set
TEST_DIR = DDP.test_dir
CLASS_LIST = DDP.class_list

for folder in CLASS_LIST:
    # predictions
    print("Predictions for images in " + folder + " class")
    path = os.path.join(TEST_DIR, folder)
    path_list = os.listdir(path)

    for fn in path_list:
        img_path = os.path.join(path, fn)
        img = image.load_img(img_path, target_size = (150, 150))

        xs = image.img_to_array(img)
        xs = np.expand_dims(xs, axis = 0)

        images = np.vstack([xs])
        classes = model.predict(images, batch_size = 20)

        for idx in range(len(classes)):
            if classes[idx] > 0.7:
                print("\n" + fn + " is a " + folder)

