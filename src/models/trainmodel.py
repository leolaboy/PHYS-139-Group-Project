#Script for training model

import tensorflow as tf

from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Flatten, Dense, Activation #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore

input_length = 1000
num_channels = 1

#Calling He_Normal Initializer
intializer = tf.keras.initializers.HeNormal()


model = Sequential()

# First block of Conv1D layers
model.add(Conv1D(filters=256, kernel_size=5, strides=1, padding='valid', 
                 kernel_initializer=intializer, input_shape=(input_length, num_channels), 
                 activation='relu',
                   name='Block 1, 1D Conv Layer 1'))
model.add(BatchNormalization(name='Batch Norm Block 1 Layer 1'))


model.add(Conv1D(filters=256, kernel_size=5, strides=1, padding='valid', 
                 kernel_initializer=intializer, activation='relu',
                   name='Block 1, 1D Conv Layer 2'))
model.add(BatchNormalization(name='Batch Norm Block 1 Layer 2'))


# Max pooling after the first block
model.add(MaxPooling1D(pool_size=5, name="Max Pooling"))

# Second block of Conv1D layers
model.add(Conv1D(filters=32, kernel_size=5, strides=1, padding='valid', 
                 kernel_initializer=intializer, activation='relu',
                   name='Block 2, 1D Conv Layer 1'))
model.add(BatchNormalization(name='Batch Norm Block 2 Layer 1'))


model.add(Conv1D(filters=32, kernel_size=5, strides=1, padding='valid', 
                 kernel_initializer=intializer, activation='relu',
                   name='Block 2, 1D Conv Layer 2'))
model.add(BatchNormalization(name='Batch Norm Block 2 Layer 2'))


# Flatten the output from the convolutional layers
model.add(Flatten(name='Flatten'))

# Fully connected layers
model.add(Dense(units=64, kernel_initializer=intializer, activation='relu', 
                name='Dense 64'))


model.add(Dense(units=8, kernel_initializer=intializer, activation='relu', 
                name='Dense 8'))

# Output layer
model.add(Dense(units=2, kernel_initializer=intializer, activation="softmax", 
                name='Dense 2'))


# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Assuming you have training data in variables X_train and y_train
# X_train should be of shape (num_samples, input_length, num_channels)
# y_train should be of shape (num_samples, 1)
# history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
