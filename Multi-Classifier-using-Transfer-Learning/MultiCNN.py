"""
MultiCNN:
    1. Defines the class Model to add the bottom layers to the pre trained InceptionV3 model
    2. Initializes the neural network model

"""
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
import PreTrainedInceptionV3 as PTI

# defining new Dense model to put on bottom of the Inception models selected last layer

LAST_OUTPUT = PTI.last_output
PRE_TRAINED_MODEL = PTI.pre_trained_model

class MODEL:

    """
    class to add bottom layers to the pre trained (InceptionV3) model
    """

    def __init__(self, pre_trained_model, last_output, dense1_neurons, dense1_activation, dense2_neurons, 
                    dense2_activation, dropout_proportion, output_neurons, output_activation):
        """
        Inputs:
            pre_trained_model: pre trained InceptionV3 model
            last_output: last layer output from the pre trained model
            dense1_neurons: number of neurons in the first fully connected dense layer
            dense1_activation: activation function of the first fully connected dense layer
            dense2_neurons: number of neurons in the second fully connected dense layer
            dense2_activation: activation function of the second fully connected dense layer
            dropout_proportion: proportion of neurons to dropout
            output_neurons: number of neurons in the output layer
            output_activation: activation function of the output layer

        Action:
            Initializes the neural network model
        """
        # Flatenning the output layer to 1 dimension
        xs = layers.Flatten()(last_output)
        # Adding a fully connected layers
        xs = layers.Dense(dense1_neurons, activation=dense1_activation)(xs)
        xs = layers.Dense(dense2_neurons, activation=dense2_activation)(xs)
        # Adding Dropout layer
        xs = layers.Dropout(dropout_proportion)(xs)
        # Adding output layer for classification
        xs = layers.Dense(output_neurons, activation=output_activation)(xs)
        
        self.xs = xs
        self.model = Model(pre_trained_model.input, self.xs)

    def get_model(self):
        """
        Returns the initialized model
        """
        return self.model

# initializing model for training
__Model__ = MODEL(PRE_TRAINED_MODEL, LAST_OUTPUT, 1024, "relu", 512, "relu", 0.3, 10, "softmax")
model = __Model__.get_model()
