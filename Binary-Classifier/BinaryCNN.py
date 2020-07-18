import tensorflow as tf

class BinaryCNN:

    def __init__(self):
        
        self.model = None

    def build_model(self, input_shape, conv1_filters, conv1_filter_size, conv1_activation, 
                conv2_filters, conv2_filter_size, conv2_activation, dropout, 
                dense_neurons, dense_activation, output_activation):

        self.model = tf.keras.models.Sequential([
            # First convolutional and Maxpooling layers
            tf.keras.layers.Conv2D(conv1_filters, conv1_filter_size, 
                    activation=conv1_activation, input_shape = input_shape),
            tf.keras.layers.MaxPooling2D(2, 2), 
            # Second convolutional and Maxpooling layers
            tf.keras.layers.Conv2D(conv2_filters, conv2_filter_size, 
                    activation=conv2_activation),
            tf.keras.layers.MaxPooling2D(2, 2),
            # Flatten to 1 dimention
            tf.keras.layers.Flatten(),
            # Droupout layer
            tf.keras.layers.Dropout(dropout),
            # First fully connected hidden layer
            tf.keras.layers.Dense(dense_neurons, activation=dense_activation),
            # Output layer for binary classification
            tf.keras.layers.Dense(1, activation=output_activation)
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