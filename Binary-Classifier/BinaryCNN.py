import tensorflow as tf

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