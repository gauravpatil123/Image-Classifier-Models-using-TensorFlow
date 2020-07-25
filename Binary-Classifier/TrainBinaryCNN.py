"""
TrainBinaryCNN:
    1. Sets the desired training accuracy [0.0, 1.0]
    2. Define myCallback class and initializes a callback per epoch for model
    3. Builds a neural network model
    4. prints the model summary
    5. Initializes the train and validation datagenerators
    6. Complies the model using a custom optimizer and loss function
    7. Trains the model / fits model on training dataset
    8. saves the model as "BinaryCNN.h5"
    9. extracts evaluation metrics from the trained model (accuracy, validation accuracy, loss, validation loss)
    10. saves comparison plots of the model
    11. tests the model on the test set
"""

import tensorflow as tf
import BinaryImageGenerators as BIG
import BinaryCNN as CNN
import PlotCode as PC
import DatasetDirectoryPreprocessing as DDP
import numpy as np
import os
import matplotlib.image as mpimg
from keras_preprocessing import image

# Desired Training accuracy
DESIRED_TRAINING_ACC = 0.999

# configuring callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > DESIRED_TRAINING_ACC):
            print("\n Reached desired traininh accuracy of "+str(DESIRED_TRAINING_ACC * 100)+ "%, so cancelling further training")
            self.model.stop_training = True

callbacks = myCallback()

# CNN model
Model = CNN.BinaryCNN()
model = Model.build_model((150, 150, 3), 64, (3, 3), 'relu', 64, (3, 3), 'relu', 0.4, 256, 'relu', 'sigmoid')

# model summary
model.summary()

# Image Generators
TRAIN_GENERATOR = BIG.train_generator
VALIDATION_GENERATOR = BIG.validation_generator

# compiling model
model.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics=['accuracy'])

# training model
history = model.fit(TRAIN_GENERATOR,
                    validation_data = VALIDATION_GENERATOR,
                    steps_per_epoch = 100, # batch size = 200, dataset size = 20000
                    epochs = 15,
                    validation_steps = 25, # batch size = 200, dataset size = 5000
                    verbose = 1,
                    callbacks = [callbacks])

# saving trained weights
model.save("binaryCNN.h5")

# plotting results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

PC.plot(acc, val_acc, epochs, 'accuracy', 'train', 'validation', 'g', 'b')
PC.plot(loss, val_loss, epochs, 'loss', 'train', 'validation', 'r', 'orange')

# testing model on test set
TEST_DIR = DDP.test_dir
TEST_LIST  = os.listdir(TEST_DIR)
print("\nEvaluating model on test Set")
print("\nTest set size = ", len(TEST_LIST))

for fn in TEST_LIST:
    # predicting images
    path = os.path.join(TEST_DIR + fn)
    #try:
    img = image.load_img(path, target_size=(150, 150))
    #except:
    #    continue

    xs = image.img_to_array(img)
    xs = np.expand_dims(xs, axis = 0)

    images = np.vstack([xs])
    classes = model.predict(images, batch_size = 20)
    #print(fn)
    #print(classes)
    if classes[0] < 0.5:
        print("\n" + fn + " is a cat")
    else:
        print("\n" + fn + " is a dog")