import tensorflow as tf

class BinaryCNN:
    """
    Class to define a learning model
    """

    def __init__(self):
        """
        Initializes model to None
        """
        
        self.model = None

    def build_model(self, input_shape, conv1_filters, conv1_filter_size, conv1_activation, 
                conv2_filters, conv2_filter_size, conv2_activation, dropout, 
                dense_neurons, dense_activation, output_activation):
        """
        Input:
            input_shape: the input shape of image tensors
            conv1_filters: number of filters for the first convolutional layer
            conv1_filter_size: custom filter shape for the first convolutional layer
            conv1_activation: custom activation function for the first convolutional layer
            conv2_filters: number of filters for the second convolutional layer
            conv2_filter_size: custom filter shape for the second convolutional layer
            conv2_activation: custom activation function for the second convolutional layer
            dropout: proportion of neurons to dropout for pruning
            dense_neurons: number of nuerons for the fully connected hidden layer
            dense_activation: activation function for the hidden layer
            output_activation: activation function for the output layer
        Output:
            model: the neural network model build using the inputs
        """

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
    
    def load_model(self, model_name):
        """
        Input:
            model_name: path/name of the model file to be loaded
        Output:
            model: the nueral network model loaded from file
        """
        self.model = tf.keras.models.load_model(model_name)
        return self.model
