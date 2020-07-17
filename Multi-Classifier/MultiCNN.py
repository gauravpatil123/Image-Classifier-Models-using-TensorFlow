import tensorflow as tf

class MultiCNN:

    model = None

    input_shape = None
    conv1_filters = None
    conv1_filter_shape = None
    conv1_activation = None
    conv2_filters = None
    conv2_filter_shape = None
    conv2_activation = None
    conv3_filters = None
    conv3_filter_shape = None
    conv3_activation = None
    conv4_filters = None
    conv4_filter_shape = None
    conv4_activation = None
    conv5_filters = None
    conv5_filter_shape = None
    conv5_activation = None
    dropout = None
    dense1_neurons = None
    dense1_activation = None
    dense2_neurons = None
    dense2_activation = None
    output_neurons = None
    output_activation = None

    def __init__(self, input_shape, conv1_filters, conv1_filter_shape, conv1_activation, 
                conv2_filters, conv2_filter_shape, conv2_activation,
                conv3_filters, conv3_filter_shape, conv3_activation,
                conv4_filters, conv4_filter_shape, conv4_activation,
                conv5_filters, conv5_filter_shape, conv5_activation,
                dropout, dense1_neurons, dense1_activation, 
                dense2_neurons, dense2_activation,
                output_neurons, output_activation):
        
        self.input_shape = input_shape
        self.conv1_filters = conv1_filters
        self.conv1_filter_shape = conv1_filter_shape
        self.conv1_activation = conv1_activation
        self.conv2_filters = conv2_filters
        self.conv2_filter_shape = conv2_filter_shape
        self.conv2_activation = conv2_activation
        self.conv3_filters = conv3_filters
        self.conv3_filter_shape = conv3_filter_shape
        self.conv3_activation = conv3_activation
        self.conv4_filters = conv4_filters
        self.conv4_filter_shape = conv4_filter_shape
        self.conv4_activation = conv4_activation
        self.conv5_filters = conv5_filters
        self.conv5_filter_shape = conv5_filter_shape
        self.conv5_activation = conv5_activation
        self.dropout = dropout
        self.dense1_neurons = dense1_neurons
        self.dense1_activation = dense1_activation
        self.dense2_neurons = dense2_neurons
        self.dense2_activation = dense2_activation
        self.output_neurons = output_neurons
        self.output_activation = output_activation

    def build_model(self):

        self.model = tf.keras.models.Sequential([
        # First Convolution layer
        tf.keras.layers.Conv2D(self.conv1_filters, self.conv1_filter_shape, 
                activation = self.conv1_activation, input_shape = self.input_shape),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Second Convolution layer
        tf.keras.layers.Conv2D(self.conv2_filters, self.conv2_filter_shape, 
                activation = self.conv2_activation),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Third Convolution layer
        tf.keras.layers.Conv2D(self.conv3_filters, self.conv3_filter_shape, 
                activation = self.conv3_activation),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Fourth Convolution layer
        tf.keras.layers.Conv2D(self.conv4_filters, self.conv4_filter_shape, 
                activation = self.conv4_activation),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Fifth Convolution layer
        tf.keras.layers.Conv2D(self.conv5_filters, self.conv5_filter_shape, 
                activation = self.conv5_activation),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten to 1 dimension
        tf.keras.layers.Flatten(),
        # Dropout layer
        tf.keras.layers.Dropout(self.dropout),
        # First fully connected hidden layer
        tf.keras.layers.Dense(self.dense1_neurons, activation = self.dense1_activation),
        # Second fully connected hidden layer
        tf.keras.layers.Dense(self.dense2_neurons, activation = self.dense2_activation),
        # Output layers
        tf.keras.layers.Dense(self.output_neurons, activation = self.output_activation)
        ])

        return self.model

# old model
"""
model = tf.keras.models.Sequential([
        # First Convolution layer
        tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Second Convolution layer
        tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Third Convolution layer
        tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Fourth Convolution layer
        tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Fifth Convolution layer
        tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten to 1 dimension
        tf.keras.layers.Flatten(),
        # Dropout layer
        tf.keras.layers.Dropout(0.3),
        # First fully connected hidden layer
        tf.keras.layers.Dense(512, activation = 'relu'),
        # Second fully connected hidden layer
        tf.keras.layers.Dense(512, activation = 'relu'),
        # Output layers
        tf.keras.layers.Dense(10, activation = 'softmax')
])
"""