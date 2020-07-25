import tensorflow as tf

class MultiCNN:
    """
    Class to define the neural network model
    """

    def __init__(self):
        """
        Initializes model to None
        """
        
        self.model = None

    def build_model(self, input_shape, conv1_filters, conv1_filter_shape, conv1_activation, 
                conv2_filters, conv2_filter_shape, conv2_activation,
                conv3_filters, conv3_filter_shape, conv3_activation,
                conv4_filters, conv4_filter_shape, conv4_activation,
                conv5_filters, conv5_filter_shape, conv5_activation,
                dropout, dense1_neurons, dense1_activation, 
                dense2_neurons, dense2_activation,
                output_neurons, output_activation):
        """
        Input:
            input_shape: the input shape of image tensor
            conv1_filters: number of filters for the first convolutional layer
            conv1_filter_shape: custom filter shape for the first convolutional layer
            conv1_activation: custom activation function for the first convolutional layer
            conv2_filters: number of filters for the second convolutional layer
            conv2_filter_shape: custom filter shape for the second convolutional layer
            conv2_activation: custom activation function for the second convolutional layer
            conv3_filters: number of filters for the third convolutional layer
            conv3_filter_shape: custom filter shape for the third convolutional layer
            conv3_activation: custom activation function for the third convolutional layer
            conv4_filters: number of filters for the fourth convolutional layer
            conv4_filter_shape: custom filter shape for the fourth convolutional layer
            conv4_activation: custom activation function for the fourth convolutional layer
            conv5_filters: number of filters for the fifth convolutional layer
            conv5_filter_shape: custom filter shape for the fifth convolutional layer
            conv5_activation: custom activation function for the fifth convolutional layer
            dropout: proportion of neurons to dropout for pruning
            dense1_neurons: number of neurons for the first fully connected hidden layer
            dense1_activation: activation function for the first fully connected hidden layer
            dense2_neurons: number of neurons for the second fully connected hidden layer
            dense2_activation: activation function for the second fully connected hidden layer
            output_neurons: number of output neurons
            output_activation: activation function for the output layer
        Output:
            model: the neural network model build using the inputs
        """

        self.model = tf.keras.models.Sequential([
        # First Convolution layer
        tf.keras.layers.Conv2D(conv1_filters, conv1_filter_shape, 
                activation = conv1_activation, input_shape = input_shape),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Second Convolution layer
        tf.keras.layers.Conv2D(conv2_filters, conv2_filter_shape, 
                activation = conv2_activation),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Third Convolution layer
        tf.keras.layers.Conv2D(conv3_filters, conv3_filter_shape, 
                activation = conv3_activation),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Fourth Convolution layer
        tf.keras.layers.Conv2D(conv4_filters, conv4_filter_shape, 
                activation = conv4_activation),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Fifth Convolution layer
        tf.keras.layers.Conv2D(conv5_filters, conv5_filter_shape, 
                activation = conv5_activation),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten to 1 dimension
        tf.keras.layers.Flatten(),
        # Dropout layer
        tf.keras.layers.Dropout(dropout),
        # First fully connected hidden layer
        tf.keras.layers.Dense(dense1_neurons, activation = dense1_activation),
        # Second fully connected hidden layer
        tf.keras.layers.Dense(dense2_neurons, activation = dense2_activation),
        # Output layers
        tf.keras.layers.Dense(output_neurons, activation = output_activation)
        ])

        return self.model

    def load_model(self, model_name):
        """
        Input:
            model_name: path/name of the model file to be loaded
        Output:
            the neural network model loaded from file
        """
        self.model = tf.keras.models.load_model(model_name)
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