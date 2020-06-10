import tensorflow as tf
import BinaryImageGenerators as BIG
import BinaryCNN as CNN
import PlotCode as PC

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
model = CNN.model

# model summary
model.summary()

# Image Generators
TRAIN_GENERATOR = BIG.train_generator
VALIDATION_GENERATOR = BIG.validation_generator

# training model
history = model.fit(TRAIN_GENERATOR,
                    validation_Data = VALIDATION_GENERATOR,
                    steps_per_epoch = # calculate according to dataset,
                    epochs = 20,
                    validation_steps = # calculate according to dataset,
                    verbose = 1,
                    calbacks = [callbacks])

# plotting results
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

PC.plot(acc, val_acc, epochs, 'accuracy', 'train', 'validation', 'g', 'b')
PC.plot(loss, val_loss, epochs, 'loss', 'train', 'validation', 'r', 'orange')