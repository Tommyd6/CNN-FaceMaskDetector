from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense,Dropout,LeakyReLU
from keras.losses import Huber
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
from matplotlib import pyplot as plt
# this section makes the layers
# the layers depth and activation function are declared when making the layers
# sequential means each layer has one tensor as input and one tensor as output (FFNN)
model =Sequential([
    Conv2D(50, (3,3), activation=LeakyReLU(alpha=0.05), input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(50, (3,3), activation=LeakyReLU(alpha=0.05)),
    MaxPooling2D(2,2),

    Conv2D(50, (3, 3), activation=LeakyReLU(alpha=0.05)),
    MaxPooling2D(2, 2),

    Flatten(),  # changes the output tensor from previous layer to a rank 1 tensor
    Dropout(0.5),  # used to avoid overfitting
    Dense(50, activation='tanh'),
    Dense(3, activation='softmax')
])
# This sections adds the ADAM optimizer, Huber loss funtion and sets the metric to be accuracy
model.compile(optimizer=Adam(learning_rate=0.0005), loss=Huber(), metrics=['accuracy'])
# This prints out the information on the layers
model.summary()
TRAINING_DIR = "./train"
# this sections is some preprocessing to training dataset
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
# this sections add the preprocessing to the data and puts it into batches
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=8,
                                                    target_size=(150, 150))

VALIDATION_DIR = "./test"
# this just rescale the pixels for test sets
validation_datagen = ImageDataGenerator(rescale=1.0/255)
# this section rescales and resizes the test data as well as putting it into batches
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                         batch_size=8,
                                                         target_size=(150, 150))
# used for saving best model
checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')

# trains the model
history = model.fit(train_generator,epochs=3,validation_data=validation_generator,callbacks=[checkpoint],shuffle=True);
# this section gets the plots of accuracy compared to epochs
plot1 = plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plot2 = plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()