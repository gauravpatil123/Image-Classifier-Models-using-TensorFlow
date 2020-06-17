import tensorflow as tf

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
        tf.keras.layers.Dense(150, activation = 'softmax')
])
