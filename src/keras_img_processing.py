import keras as keras
import keras_preprocessing.image as processing
import numpy as np
from sklearn.utils import class_weight

data_path = 'E:\\Data\\train_val2018\\TrainVal\\'
batch_size = 128

datagen = processing.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
x_batch_list = []
y_batch_list = []
data_generator = datagen.flow_from_directory(
    data_path,
    target_size=(28, 28),
    batch_size=batch_size,
    class_mode='categorical')
y_train = data_generator.classes
class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(y_train),
                                                  y_train)
print('')
