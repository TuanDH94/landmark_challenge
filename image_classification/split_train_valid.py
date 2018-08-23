import os
import shutil
import numpy as np
from os import listdir
from os.path import isfile, isdir, join

file_paths = 'E:\\Data\\train_val2018\\TrainVal\\'
dest11 = "E:\\Data\\train_val2018\\Valid\\"

all_folders = [f for f in listdir(file_paths) if isdir(join(file_paths, f))]
for folder in all_folders:
    full_path_folder = join(file_paths, folder)
    full_path_folder_valid = join(dest11, folder)
    if not os.path.exists(full_path_folder_valid):
        os.makedirs(full_path_folder_valid)
    all_files = [f for f in listdir(full_path_folder) if isfile(join(full_path_folder, f))]
    for file in all_files:
        full_path_file = join(full_path_folder, file)
        full_path_file_valid = join(full_path_folder_valid, file)
        if np.random.rand(1) < 0.2:
            shutil.move(full_path_file, full_path_file_valid)
