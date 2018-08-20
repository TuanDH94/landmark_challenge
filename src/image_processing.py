import keras as kr
import numpy as np
import cv2
from keras_preprocessing.image import img_to_array
from os import listdir
from os.path import isfile, isdir, join
from sklearn.model_selection import train_test_split

def append_img_to_data(data, image, labels, data_label):
    image = cv2.resize(image, (28, 28))
    image = img_to_array(image)
    data.append(image)
    labels.append(int(data_label))
def read_data_from_file(file_paths):
    data = []
    labels = []
    all_folders = [f for f in listdir(file_paths) if isdir(join(file_paths, f))]
    for folder in all_folders:
        full_path_folder = join(file_paths, folder)
        all_files = [f for f in listdir(full_path_folder) if isfile(join(full_path_folder, f))]
        for file in all_files:
            full_path_file = join(full_path_folder, file)
            image = cv2.imread(full_path_file)
            if image is not None:
               append_img_to_data(data, image, labels, folder)
        print("Processing " + folder + " done!")
    x = np.array(data)
    y = np.asarray(labels).T
    print("processing done!")
    return data, labels


def split_data(data, labels):
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
    return (trainX, trainY),(testX, testY)


def load_data(file_paths):
    data, labels = read_data_from_file(file_paths)
    return split_data(data, labels)

if __name__ == '__main__':
    mypath = 'E:\\Data\\train_val2018\\TrainVal\\'
    load_data(mypath)