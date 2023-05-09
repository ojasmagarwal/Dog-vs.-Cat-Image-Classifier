#importing the modules

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

#PART 1 - DATA PREPROCESSING

#Preprocessing the training set

train_datagen = ImageDataGenerator( #image augmentation - we apply transformations to the images to avoid overfitting, we apply transformations like zooming in and zooming out,horizontal flip,rotations etc. we just apply these transformations to the training set
                rescale=1./255,   #feature scaling
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

training_set= train_datagen.flow_from_directory(  #flow from directory connects the image augmentation tool to the database
            'dataset/training_set',
            target_size=(64,64), #resizing the images so that pc takes less time
            batch_size=32,
            class_mode='binary') #if more than 2 then write 'categorical'

#Preprocessing the test set

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
            'dataset/test_set',
            target_size=(64,64),
            batch_size=32,
            class_mode='binary')

#PART 2 - BUILDING THE CNN

#initialising the cnn

cnn = tf.keras.models.Sequential()

#Step 1 - COnvolution

cnn.add(tf.keras.layers.Conv2D(filters =32,kernel_size=3,activation = 'relu',input_shape=[64,64,3]))   #adds a convolutional layer
#kernel size is the size of the feature detector, input shape is resized to the size we did at data preprocessing and the last parameter is 3 because the images is colored, if the images were black and white the value would be 1

#Step 2 - Pooling

cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2,strides = 2))

#Adding a second convolutional layer

cnn.add(tf.keras.layers.Conv2D(filters =32,kernel_size=3,activation = 'relu'))  # we remove the input shape here because the cnn already has input shape in the first layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2,strides = 2))

#Step 3 - Flattening

cnn.add(tf.keras.layers.Flatten())

#Step 4 - Full Connection

cnn.add(tf.keras.layers.Dense(units = 128,activation='relu'))

#Step 5 - Output Layer

cnn.add(tf.keras.layers.Dense(units = 1,activation='sigmoid'))

#PART 3 - TRAINING THE CNN

#Compiling the CNN

cnn.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])

#Training the CNN on the training set and evaluating it on the test set

cnn.fit(x=training_set,validation_data=test_set,epochs = 15)

#STEP 4 - MAKING A SINGLE PREDICTION

import keras.utils as image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis = 0)
result = cnn.predict(test_image)
training_set.class_indices #to know which no represents dog or cat (0 or 1)
if result[0][0] == 1: #first accessing the batch then the single element of the batch
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)
