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
            print("\n Reached desired training accuracy of "+str(DESIRED_TRAINING_ACC * 100)+ "%, so cancelling further training")
            self.model.stop_training = True

callbacks = myCallback()

# CNN model
Model = CNN.MultiCNN((300, 300, 3), 32, (3, 3), 'relu',
                     64, (3, 3), 'relu', 64, (3, 3), 'relu',
                     128, (3, 3), 'relu', 128, (3, 3), 'relu',
                     0.3, 512, 'relu', 512, 'relu', 10, 'softmax')
model = Model.build_model()

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
                    epochs = 100, 
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

PC.plot(acc, val_acc, epochs, 'accuracy', 'train', 'validation', 'g', 'b')
PC.plot(loss, val_loss, epochs, 'loss', 'train', 'validation', 'r', 'orange')
