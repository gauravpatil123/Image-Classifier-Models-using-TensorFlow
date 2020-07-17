import tensorflow as tf

class BinaryCNN:

    model = None

    input_shape = None
    conv1_filters = None
    conv1_filter_size = None
    conv1_activation = None
    conv2_filters = None
    conv2_filter_size = None
    conv2_activation = None
    dropout = None
    dense_neurons = None
    dense_activation = None
    output_activation = None

    def __init__(self, input_shape, conv1_filters, conv1_filter_size, conv1_activation, 
                 conv2_filters, conv2_filter_size, conv2_activation, dropout, 
                 dense_neurons, dense_activation, output_activation):
        
        self.input_shape = input_shape
        self.conv1_filters = conv1_filters
        self.conv1_filter_size = conv1_filter_size
        self.conv1_activation = conv1_activation
        self.conv2_filters = conv2_filters
        self.conv2_filter_size = conv2_filter_size
        self.conv2_activation = conv2_activation
        self.dropout = dropout
        self.dense_neurons = dense_neurons
        self.dense_activation = dense_activation
        self.output_activation = output_activation

    def build_model(self):

        self.model = tf.keras.models.Sequential([
            # First convolutional and Maxpooling layers
            tf.keras.layers.Conv2D(self.conv1_filters, self.conv1_filter_size, 
                    activation=self.conv1_activation, input_shape = self.input_shape),
            tf.keras.layers.MaxPooling2D(2, 2), 
            # Second convolutional and Maxpooling layers
            tf.keras.layers.Conv2D(self.conv2_filters, self.conv2_filter_size, 
                    activation=self.conv2_activation),
            tf.keras.layers.MaxPooling2D(2, 2),
            # Flatten to 1 dimention
            tf.keras.layers.Flatten(),
            # Droupout layer
            tf.keras.layers.Dropout(self.dropout),
            # First fully connected hidden layer
            tf.keras.layers.Dense(self.dense_neurons, activation=self.dense_activation),
            # Output layer for binary classification
            tf.keras.layers.Dense(1, activation=self.output_activation)
        ])

        return self.model


# old model
""" 
model = tf.keras.models.Sequential([
    # First convolutional and Maxpooling layers
    tf.keras.layers.Conv2D(64,(3, 3), activation='relu', input_shape = (150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2), 
    # Second convolutional and Maxpooling layers
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Third convolutioal and Maxpooling layers
    #tf.keras.layers.Conv2D(64, (3, 3),  activation='relu'),
    #tf.keras.layers.MaxPooling2D(2, 2),
    # Fourth convolutional and Maxpooling layers
    #tf.keras.layers.Conv2D(128, (3, 3),  activation='relu'),
    #tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten to 1 dimention
    tf.keras.layers.Flatten(),
    # Droupout layer
    tf.keras.layers.Dropout(0.4),
    # First fully connected hidden layer
    tf.keras.layers.Dense(256, activation='relu'),
    # Second full connected hiddenlayer
    #tf.keras.layers.Dense(512, activation='relu'),
    # Output layer for binary classification
    tf.keras.layers.Dense(1, activation='sigmoid')
])
"""