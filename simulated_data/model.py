import json
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Data loading and preprocessing
with open('sowed_data.json', 'r') as f:
    raw_data = json.load(f)

CLASSES = 6

def convert_data_point(data_point):
    wl = np.array(data_point["wl"])
    r = np.array(data_point["r"])
    c = np.array([item[2] for item in data_point["l"]])

    #print(c)

    if c.tolist().index(1) <= CLASSES:
        #print('out')
        c = c[:CLASSES]
    else:
        return np.array(False), False, False

    #print(c)

    return wl, r, c

X, y, wl = [], [], None
for data_point in raw_data:
    wl_, r, c = convert_data_point(data_point)
    if wl_.any() == False:
        continue
    X.append(r)
    new_c = [1 if i > 0.5 else 0 for i in c]
    y.append(new_c)
    wl = wl_

X = np.array(X)
y = np.array(y)
X = preprocessing.normalize(X)
y = preprocessing.normalize(y)

print(X.shape)

# 42 86

# 65 925

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

#flatten, 64, 128, 64 relu, softmax

def create_model(input_shape, num_classes):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(64,),
        #Dense(256,),
        #Dense(512),
        Dense(128,),
        Dense(64,),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Initialize the custom model
input_shape = (X.shape[1],)
num_classes = y.shape[1]
model = create_model(input_shape, num_classes)
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Training loop
print("Starting training")
num_epochs = 60
best_val_loss = float('inf')

history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=64, 
                    validation_data=(X_test, y_test), verbose=1)

for epoch in range(num_epochs):
    val_loss = history.history['val_loss'][epoch]
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model.save('best_model.keras')

model.save('last_model.keras')

model = tf.keras.models.load_model('best_model.keras')
train_accuracy = model.evaluate(X_train, y_train, verbose=0)[1]
test_accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
print(f"Best Model, Training Accuracy: {train_accuracy}, Testing Accuracy: {test_accuracy}")