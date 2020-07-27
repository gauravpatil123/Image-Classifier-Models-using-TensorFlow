"""
MultiCNN:
    Adds bottom layers to the pretrained (Inception) model
"""
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
import PreTrainedInceptionV3 as PTI

# defining new Dense model to put on bottom of the Inception models selected last layer

LAST_OUTPUT = PTI.last_output
PRE_TRAINED_MODEL = PTI.pre_trained_model

# Flatenning the output layer to 1 dimension
xs = layers.Flatten()(LAST_OUTPUT)
# Adding Dropout layer
#xs = layers.Dropout(0.2)(xs)
# Adding a fully connected layers
xs = layers.Dense(1024, activation="relu")(xs)
xs = layers.Dense(512, activation="relu")(xs)
# Adding Dropout layer
xs = layers.Dropout(0.3)(xs)
# Adding output layer for classification
xs = layers.Dense(10, activation="softmax")(xs)

# defining model object
model = Model(PRE_TRAINED_MODEL.input, xs)