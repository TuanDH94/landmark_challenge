from keras.applications.vgg16 import VGG16
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras.backend as K
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, isdir, join
def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def detect_error_file():
    file_paths = 'E:\\Data\\public_test2018\\Public\\'
    all_files = [f for f in listdir(file_paths) if isfile(join(file_paths, f))]
    for file in all_files:
        full_path_file = join(file_paths, file)
        try:
            img = Image.open(full_path_file)  # open the image file
            img.verify()  # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
            print('Bad file:', full_path_file)


if __name__ == '__main__':
#     img_rows, img_cols = 28, 28
#     num_classes = 10
#
#     # the data, split between train and test sets
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
#     if K.image_data_format() == 'channels_first':
#         x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#         x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#         input_shape = (1, img_rows, img_cols)
#     else:
#         x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#         x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#         input_shape = (img_rows, img_cols, 1)
#
#     x_train = x_train.astype('float32')
#     x_test = x_test.astype('float32')
#     x_train /= 255
#     x_test /= 255
#     print('x_train shape:', x_train.shape)
#     print(x_train.shape[0], 'train samples')
#     print(x_test.shape[0], 'test samples')
#
#     # convert class vectors to binary class matrices
#     y_train = keras.utils.to_categorical(y_train, num_classes)
#     y_test = keras.utils.to_categorical(y_test, num_classes)
#
#     model_path = 'E:\\PythonSource\\keras_practice\\image_classification\\cnn_minst_model_weight.h5'
#     model = create_model(input_shape=input_shape, num_classes=num_classes)
#     model.load_weights(model_path)
# # predict results
#     results = model.predict(x_test)
#
# # select the indix with the maximum probability
#     top_k_result = results.argsort()[:, :3][::-1]
#     top_k = K.in_top_k(results, y_test, 3)
#     print()
    detect_error_file()