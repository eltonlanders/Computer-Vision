# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 20:09:22 2020

@author: elton
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras

#Load MNIST data using built-in datasets download function
mnist=tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalize the pixel values by dividing each pixel by 255
x_train, x_test = x_train/255.0, x_test/255.0

#Build the 4-layer neural network (MLP)
model=keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(60, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
    ])

#Compile the model and set optimizer,loss function and metrics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Finally, train or fit the model
trained_model = model.fit(x_train, y_train, validation_split=0.3, epochs=100)

#Visualize loss  and accuracy history
plt.plot(trained_model.history['loss'], 'r--')
plt.plot(trained_model.history['accuracy'], 'b-')
plt.legend(['Training Loss', 'Training Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Percent')
plt.show()

#Evaluate the result using the test set.
evalResult = model.evaluate(x_test, y_test, verbose=1)
print("Evaluation", evalResult)
predicted = model.predict(x_test)
print("Predicted", predicted)

confusion = tf.math.confusion_matrix(y_test, np.argmax(predicted,
                                                         axis=1),
                                       num_classes=10)
tf.print(confusion)



