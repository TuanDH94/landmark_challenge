import glob
import os
import librosa
import librosa.display
import librosa.core
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
from os import listdir
from os.path import isfile, isdir, join
import pandas as pd

def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X, sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds


def plot_waves(sound_names, raw_sounds):
    i = 1
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(10, 1, i)
        librosa.display.waveplot(np.array(f), sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 1: Waveplot', x=0.5, y=0.915, fontsize=18)
    plt.show()


def plot_specgram(sound_names, raw_sounds):
    i = 1
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(10, 1, i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 2: Spectrogram', x=0.5, y=0.915, fontsize=18)
    plt.show()

def plot_log_power_specgram(sound_names,raw_sounds):
    i = 1
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        D = librosa.core.amplitude_to_db(np.abs(librosa.stft(f))**2, ref_power=np.max)
        librosa.display.specshow(D,x_axis='time' ,y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 3: Log power spectrogram',x=0.5, y=0.915,fontsize=18)
    plt.show()

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)

   # features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
    features = pd.DataFrame({
        'mfccs' : mfccs,
        'chroma' : chroma,
        'mel' : mel,
        'constrast' : contrast,
        'tonnetz' : tonnetz
    })
    return features

def processing(directory):
    all_folders = [f for f in listdir(directory) if isdir(join(directory, f))]
    _data, _gender, _region = [], [], []
    for folder in all_folders:
        full_path_folder = join(directory, folder)
        all_files = [f for f in listdir(full_path_folder) if isfile(join(full_path_folder, f))]
        for file in all_files:
            full_path_file = join(full_path_folder, file)
            

sound_path_files = ['E:\\Data\\train_voice\\debug\\female_central\\voice_1.wav']
raw_sounds = load_sound_files(sound_path_files)
sound_names = ['male']
# plot_waves(sound_names,raw_sounds)
# plot_specgram(sound_names,raw_sounds)
# plot_log_power_specgram(sound_names,raw_sounds)
features = extract_feature(sound_path_files[0])