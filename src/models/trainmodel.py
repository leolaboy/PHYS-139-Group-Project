#Script for training model

import tensorflow as tf


from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Flatten, Dense, Activation #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore
from tensorflow import keras

input_length = 1000
num_channels = 1

#Calling He_Normal Initializer
intializer = tf.keras.initializers.HeNormal()


model = Sequential()

# First block of Conv1D layers
model.add(Conv1D(filters=256, kernel_size=5, strides=1, padding='valid', 
                 kernel_initializer=intializer, input_shape=(input_length, num_channels), 
                 activation='relu',
                   name='Block_1_1D_Conv_Layer_1'))
model.add(BatchNormalization(name='Batch_Norm_Block_1_Layer_1'))


model.add(Conv1D(filters=256, kernel_size=5, strides=1, padding='valid', 
                 kernel_initializer=intializer, activation='relu',
                   name='Block_1_1D_Conv_Layer_2'))
model.add(BatchNormalization(name='Batch_Norm_Block_1_Layer_2'))


# Max pooling after the first block
model.add(MaxPooling1D(pool_size=5, name="Max_Pooling"))

# Second block of Conv1D layers
model.add(Conv1D(filters=32, kernel_size=5, strides=1, padding='valid', 
                 kernel_initializer=intializer, activation='relu',
                   name='Block_2_1D_Conv_Layer_1'))
model.add(BatchNormalization(name='Batch_Norm_Block_2_Layer_1'))


model.add(Conv1D(filters=32, kernel_size=5, strides=1, padding='valid', 
                 kernel_initializer=intializer, activation='relu',
                   name='Block_2_1D_Conv_Layer_2'))
model.add(BatchNormalization(name='Batch_Norm_Block_2_Layer_2'))


# Flatten the output from the convolutional layers
model.add(Flatten(name='Flatten'))

# Fully connected layers
model.add(Dense(units=64, kernel_initializer=intializer, activation='relu', 
                name='Dense_64'))


model.add(Dense(units=8, kernel_initializer=intializer, activation='relu', 
                name='Dense_8'))

# Output layer
model.add(Dense(units=2, kernel_initializer=intializer, activation="softmax", 
                name='Dense_2'))



# Compile the model
#Implementing the learning rate schedule makes the model crash so I took it out for now.
# lr_schedule = tf.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=100, decay_rate=0.8, staircase=True)

model.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
summary = model.summary()
