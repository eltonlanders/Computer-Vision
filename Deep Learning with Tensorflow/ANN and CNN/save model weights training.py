import tensorflow as tf
import matplotlib.pyplot as plt
import os

# The file path where checkpoint will be saved.
checkpoint_path = "cv_checkpoint_dir/mnist_model.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights.
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Load MNIST data using built-in datasets download function.
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Noramalize the pixel values by deviding each pixel by 255.
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the ANN with 4-layers.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(60, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model and set optimizer,loss function and metrics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Finally, train or fit the model, pass callbacks to save the model weights.
trained_model = model.fit(x_train, y_train, validation_split=0.3, epochs=10,
                          callbacks=[cp_callback])

# Visualize loss  and accuracy history
plt.plot(trained_model.history['loss'], 'r--')
plt.plot(trained_model.history['accuracy'], 'b-')
plt.legend(['Training Loss', 'Training Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Percent')
plt.show();

# Evaluate the result using the test set.
evalResult = model.evaluate(x_test,  y_test, verbose=1)
print("Evaluation Result: ", evalResult)

"""#Save the model weights manually
checkpoint_path = "cv_checkpoint_dir/mnist_model.ckpt" 
model.save_weights(checkpoint_path)

# Load saved weights and retrain the model
model.load_weights(checkpoint_path)

#Saving the entire model
model.save('mv_ann_model.h5') #can also pass the absolute path

#Load and create exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('mv_ann_model.h5')

# Show the model architecture
model.summary()

#Retrain the model
retrained_model = model.fit(x_train, y_train, validation_split=0.3, epochs=10)

# Load and create the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('mv_ann_model.h5')

# Predict the class of the input image from the loaded model 
predicted = model.predict(x_pixel_data) 
print("Predicted", predicted)
"""