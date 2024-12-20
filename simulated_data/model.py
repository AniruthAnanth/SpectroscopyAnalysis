import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, general_gaussian
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import random

random.seed(42)

def pre_process_sample(spectrum):
    def modified_z_score(ys):
        ysb = np.diff(ys) # Differentiated intensity values
        median_y = np.median(ysb) # Median of the intensity values
        median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ysb]) # median_absolute_deviation of the differentiated intensity values
        modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y for y in ysb] # median_absolute_deviationmodified z scores
        return modified_z_scores
        
    # The next function calculates the average values around the point to be replaced.
    def fixer(y,ma):
        threshold = 7 # binarization threshold
        spikes = abs(np.array(modified_z_score(y))) > threshold
        y_out = y.copy()
        for i in np.arange(len(spikes)):
            if spikes[i] != 0:
                w = np.arange(i-ma,i+1+ma)
                we = w[spikes[w] == 0]
                y_out[i] = np.mean(y[we])
        return y_out

    despiked_spectrum = fixer(spectrum, ma=10)
    w, p = 9, 2

    smoothed_spectrum = savgol_filter(despiked_spectrum, w, polyorder = p, deriv=0)

    return smoothed_spectrum

data = json.loads(open('data.json', 'r').read())

x = np.array(data['spectrums'])
Y = np.array(data['concentrations'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=0.2, random_state=32)

# Initialize the TensorFlow model
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(20)  # Output layer for 20 components
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Define the early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model and save the training history
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Predict on the test set
y_pred = model.predict(X_test)

print(y_test, y_pred)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot the results
plt.figure()
plt.plot(y_test, y_pred, 'o')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted Values')
plt.show()

# Plot the training history
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()