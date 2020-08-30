"""
PreTrainedInceptionV3:
    1. Defines the class PreTrainedModel to initialize the pre trained model
    2. Initializes the pre trained model and the last layer output
"""
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
import logging

class PreTrainedModel:

    def __init__(self, local_weights_file_dir, input_shape, last_layer_choice_InceptionV3):
        """
        Input:
            local_weights_file_dir: local directory of weights file for pre trained InceptionV3 model
            input_shape: custom input shapes of images
            last_layer_choice_InceptionV3: choice of last layer in the InceptionV3 model

        Action:
            1. Initializes the pre trained model 
            2. Freezes all the layers to non tarinable
            3. Initializes the selected last layer
            4. Prints model summary
            5. Initializes the output of the last layer
            6. logs the last layer shape
        """
        self.pre_trained_model = InceptionV3(input_shape = (300, 300, 3),
                                             include_top = False,
                                             weights = None)
        self.pre_trained_model.load_weights(local_weights_file_dir)

        for layer in self.pre_trained_model.layers:
            layer.trainable = False

        self.pre_trained_model.summary()

        self.last_layer = self.pre_trained_model.get_layer(last_layer_choice_InceptionV3)

        logging.basicConfig(format="%(message)s", level = logging.INFO)
        message = 'last layer output shape: ' + str(self.last_layer.output_shape)
        self.last_output = self.last_layer.output
        logging.info(message)

    def get_pre_trained_model(self):
        """
        Returns the initialized pre trained model
        """
        return self.pre_trained_model

    def get_last_output(self):
        """
        Returns the 
        """
        return self.last_output

pre_trained = PreTrainedModel('pre-trained-model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', (300, 300, 3), 'mixed8')
pre_trained_model = pre_trained.get_pre_trained_model()
last_output = pre_trained.get_last_output()

"""
# pre-trained Inception model and weights
local_weights_file = 'pre-trained-model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

# defining Inception model and removing top dense layers
pre_trained_model = InceptionV3(input_shape = (300, 300, 3),
                                include_top = False,
                                weights = None)

# loading pretrained weights into the model
pre_trained_model.load_weights(local_weights_file)

# freezing layers in pre_trained_model
for layer in pre_trained_model.layers:
    layer.trainable = False

# pre-trained model summary for output player selection
pre_trained_model.summary()

# selecting last layer of Inception model for our use
last_layer = pre_trained_model.get_layer('mixed8')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output
"""
