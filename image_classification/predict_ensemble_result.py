import keras
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Dropout, Activation, Flatten
from keras.applications import InceptionResNetV2
from keras.optimizers import SGD
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import keras_preprocessing.image as processing
import numpy as np
from sklearn.utils import class_weight
import csv
import cv2


def create_model(input_shape, num_classes):
    model = Sequential()
    # create the base pre-trained model

    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.summary()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    # # # let's add a fully-connected layer
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    return model


if __name__ == '__main__':
    processing.ImageDataGenerator(rescale=1. / 255)
    img_height, img_width = 224, 224
    input_shape = (img_height, img_width, 3)
    num_classes = 103
    test_path = 'E:\\Data\\image_debug\\'
    model_weight_path = 'E:\\PythonSource\\inception_landmark\\keras_landmark_inception_resnet_v2_model.18-0.92.h5 '
    batch_size = 32

    # init class map index
    class_str = []
    for i in range(num_classes):
        class_str.append(str(i))
    class_str = sorted(class_str)

    # init

    # load model
    model = create_model(input_shape=input_shape, num_classes=num_classes)
    model.load_weights(model_weight_path)

    # test generator
    datagen = processing.ImageDataGenerator(rescale=1. / 255)
    test_generator = datagen.flow_from_directory(
        test_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        classes=None,
        shuffle=False
    )
    # img_test = processing.load_img('E:\\Sources\\PycharmProjects\\Data\\Zalo\\Landmark\\TrainVal\\15\\10895.jpg', )
    # img = cv2.imread('E:\\Sources\\PycharmProjects\\Data\\Zalo\\Landmark\\TrainVal\\2\\15555.jpg')
    # img = cv2.resize(img, (img_width, img_height))
    # img = np.reshape(img, [1, img_width, img_height, 3])

    test_result = model.predict_generator(test_generator, workers=4)
    # print()
    test_result_top_3 = []
    with open('submission.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['id', 'predicted'])
        for i in range(test_result.shape[0]):
            # get id
            file_name = str(test_generator.filenames[i])
            id = file_name.replace('Public\\', '').replace('.jpg', '')
            predicted = ''
            for j in range(num_classes):
                index = int(class_str.index(str(j)))
                label_predicted = test_result[i][index]
                predicted = predicted + str(label_predicted) + ','
            predicted = predicted[:-1]
            writer.writerow([id, predicted])
        print('')
