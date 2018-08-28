import numpy as np
import librosa
import math
import re
from keras.utils import to_categorical
from os import listdir
from os.path import isfile, isdir, join
from keras.utils import to_categorical
from multiprocessing import Pool
import time


gender_dict = {'female': 0, 'male': 1}
region_dict = {'north': 0, 'central': 1, 'south': 2}
print(gender_dict['female'])


def getfeature(fname):
    timeseries_length = 128
    hop_length = 512
    data = np.zeros((timeseries_length, 33), dtype=np.float64)

    y, sr = librosa.load(fname)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)

    filelength = timeseries_length if mfcc.shape[1] >= timeseries_length else mfcc.shape[1]

    data[-filelength:, 0:13] = mfcc.T[0:timeseries_length, :]
    data[-filelength:, 13:14] = spectral_center.T[0:timeseries_length, :]
    data[-filelength:, 14:26] = chroma.T[0:timeseries_length, :]
    data[-filelength:, 26:33] = spectral_contrast.T[0:timeseries_length, :]

    return data


def get_instance(full_path_file_folder):
    full_path_file = full_path_file_folder[0]
    folder = full_path_file_folder[1]
    gender, region = folder.split('_')
    data = getfeature(full_path_file)
    return data, gender, region


def processing(folder_train):
    all_folders = [f for f in listdir(folder_train) if isdir(join(folder_train, f))]
    _data, _gender, _region = [], [], []
    for folder in all_folders:
        full_path_folder = join(folder_train, folder)
        all_files = [f for f in listdir(full_path_folder) if isfile(join(full_path_folder, f))]
        for file in all_files:
            full_path_file = join(full_path_folder, file)
            data, gender, region = get_instance([full_path_file, folder])
            _data.append(data)
            _gender.append(gender_dict[gender])
            _region.append(region_dict[region])
    X = np.array(_data)
    Ygender = to_categorical(np.array(_gender))
    YRegion = to_categorical(np.array(_region))
    return X, Ygender, YRegion

def multi_processing(folder_train):
    _data, _gender, _region = [], [], []
    p = Pool(4)
    full_path_file_folder_list = []
    all_folders = [f for f in listdir(folder_train) if isdir(join(folder_train, f))]
    for folder in all_folders:
        full_path_folder = join(folder_train, folder)
        all_files = [f for f in listdir(full_path_folder) if isfile(join(full_path_folder, f))]
        for file in all_files:
            full_path_file = join(full_path_folder, file)
            full_path_file_folder_list.append([full_path_file, folder])
    data_labels = p.map(get_instance, full_path_file_folder_list)
    for i in range(len(data_labels)):
        _data.append(data_labels[i][0])
        _gender.append(gender_dict[data_labels[i][1]])
        _region.append(region_dict[data_labels[i][2]])
    X = np.array(_data)
    Ygender = to_categorical(np.array(_gender))
    YRegion = to_categorical(np.array(_region))
    return X, Ygender, YRegion

if __name__ == '__main__':
    start_time = time.time()

    folder_train = 'E:\\Data\\train_voice\\debug'
    multi_processing(folder_train)
    elapsed_time = time.time() - start_time
    print(elapsed_time)