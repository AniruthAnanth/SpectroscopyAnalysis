import numpy as nps
import tensorflow as tf
import visualkeras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

num_points = 500

# Train denoising autoencoder
autoencoder = Sequential([
    Dense(16, activation='relu', input_shape=(num_points,)),
    Dense(8, activation='relu'),
    Dense(2, activation='relu'),
    Dense(8, activation='relu'),
    Dense(16, activation='relu'),
    Dense(num_points, activation='relu')
])

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.load_weights('models/autoencoder.keras')

visualkeras.graph_view(autoencoder, to_file='autoencoder.png')

# Build a TensorFlow Neural Network
regressor = Sequential([
    Dense(128, activation='relu', input_shape=(num_points,)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(15, activation='linear')  # M outputs, one for each compound concentration
])

regressor.compile(optimizer='adam', loss='mse', metrics=['mae'])

regressor.load_weights('models/v7.keras')

visualkeras.graph_view(regressor, to_file='regressor.png')
visualkeras.layered_view(regressor, to_file='regressor_layer.png')