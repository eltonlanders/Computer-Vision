import numpy as np
import pathlib
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


#Loading images from directories for training and test
trainig_img_dir ="images/chest_xray/train"
test_img_dir ="images/chest_xray/test"

# ImageDataGenerator class provides mechanism to load both small and large dataset.
# Instruct ImageDataGenerator to scale to normalize pixel values to range (0, 1)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
#Create training image iterator that will be loaded in small batch size.
#Resize all images to a standard sizes.
train_it = datagen.flow_from_directory(trainig_img_dir, batch_size=2,
                                       target_size=(512,512))
#Create training image iterator that will be loaded in small batch size.
#Resize all images to a standard sizes.
test_it = datagen.flow_from_directory(test_img_dir, batch_size=2,
                                      target_size=(512, 512))
 



# Lines 22 through 24 are optional to explore images.
# Notice, next() function call returns both pixel and labels values as numpy arrays.
train_images, train_labels = train_it.next()
test_images, test_labels = test_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (train_images.shape,
                                              train_images.min(),
                                              train_images.max()))

# Build CNN network and train with training dataset.
# You could pass argument parameters to build_cnn() function to set some of the values
# such as number of filters, strides, activation function, number of layers etc.
def build_cnn():
    model =  tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                                     strides=(2,2),
                                     input_shape=(512, 512, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), strides=(2,2), 
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), strides=(2,2), 
                                     activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    return model

# Build CNN model
model = build_cnn()
#Compile the model with optimizer and loss function
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model. fit_generator() function iteratively loads large number of images in batches
history = model.fit_generator(train_it, epochs=10, steps_per_epoch=2000,
                    validation_data=test_it, validation_steps=2)

# Save the CNN model to disk for later use.
model_path = "models/pneumiacnn"
model.save(filepath=model_path)

# Section 4: Display evaluation metrics
print(history.history.keys())
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')

plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)
