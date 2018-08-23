from os import listdir
from PIL import Image
from os import listdir
from os.path import isfile, isdir, join
import os

file_paths = 'E:\\Data\\train_val2018\\TrainVal\\'
all_folders = [f for f in listdir(file_paths) if isdir(join(file_paths, f))]
for folder in all_folders:
    full_path_folder = join(file_paths, folder)
    all_files = [f for f in listdir(full_path_folder) if isfile(join(full_path_folder, f))]
    for file in all_files:
        full_path_file = join(full_path_folder, file)
        try:
            img = Image.open(full_path_file)  # open the image file
            img.verify()  # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
            print('Bad file:', full_path_file)  # print out the names of corrupt files
            os.remove(full_path_file)


