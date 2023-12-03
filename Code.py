import numpy as np
import random
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.metrics import categorical_crossentropy
train_labels = []
train_samples = []
for i in range(1000): #95% of the data
  random_young = randint(13, 64)
  train_samples.append(random_young)
  train_labels.append(1)

  random_older = randint(65, 100)
  train_samples.append(random_older)
  train_labels.append(0)

for i in range(50): #5% of the data
  random_young = randint(13,64)
  train_samples.append(random_young)
  train_labels.append(0)

  random_older = randint(65, 100)
  train_samples.append(random_older)
  train_labels.append(1)
train_labels = np.array(train_labels) #coinverting the data into numpy arrays
train_samples = np.array(train_samples)
train_labels, train_samples = shuffle(train_labels, train_samples)

#for simplicity of the training process, we rescale the data from 13- 100 to 0 - 1
scaler = MinMaxScaler(feature_range=(0,1))
scaler_trian_samples = scaler.fit_transform(train_samples.reshape(-1,1))
model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'), #input layers = 16 (optional as in this case. Usually depends on the use case)
    Dense(units=32, activation='relu'), #hidden layers = 32
    Dense(units=2, activation='softmax') #yes or no output
    #softmax takes care of the model output functions to add up to 1
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=scaler_trian_samples, y=train_labels,validation_split = 0.2, batch_size=10, epochs=100, shuffle=True, verbose=2)
#Prediction
test_samples = []
test_labels = []
for i in range(200): #95% of the data
  random_young = randint(13, 64)
  test_samples.append(random_young)
  test_labels.append(1)

  random_older = randint(65, 100)
  test_samples.append(random_older)
  test_labels.append(0)

for i in range(10): #5% of the data
  random_young = randint(13,64)
  test_samples.append(random_young)
  test_labels.append(0)

  random_older = randint(65, 100)
  test_samples.append(random_older)
  test_labels.append(1)
test_samples = np.array(test_samples)
test_labels = np.array(test_labels)
test_samples, test_labels = shuffle(test_samples, test_labels)
scaler_test_samples = scaler.fit_transform(test_samples.reshape(-1, 1))
predictions = model.predict(x=scaler_test_samples, batch_size = 10, verbose =0)
print("Predictions: ")
for i in predictions:
  print(i)
rounded_predictions = np.argmax(predictions, axis = -1)
print("Rounded Predctions :")
for i in rounded_predictions:
  print(i) # 0 - no , 1 - yes
