"""
PreTrainedInceptionV3:
    1. loads the pretrained InceptionV3 model from locally saved weights file
    2. Set the top layers of InceptionV3 model to untrainable
    3. prints the model summary
    4. sets the 'mixed8' layer from the model as the last output layer to pass to 
       custom added bottom layers when training
"""
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3

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