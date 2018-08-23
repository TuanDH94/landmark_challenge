import keras
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, Activation, MaxPooling2D, Flatten
from keras import backend as K
import keras_preprocessing.image as processing
import numpy as np
from sklearn.utils import class_weight

import os


batch_size = 256
num_classes = 103
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_landmark_trained_model.h5'
file_path = 'E:\\Data\\train_val2018\\TrainVal\\'
img_height = 112
img_width = 112

model = Sequential()

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

for layer in base_model.layers:
    model.add(layer)
model.add(GlobalAveragePooling2D())
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# let's add a fully-connected layer
model.add(Dense(1024, activation='relu'))
# and a logistic layer -- let's say we have 200 classes
model.add(Dense(num_classes, activation='softmax'))

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

datagen = processing.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
train_generator = datagen.flow_from_directory(
    file_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

y_train = train_generator.classes
class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(y_train),
                                                  y_train)

model.fit_generator(train_generator, class_weight=class_weights)
