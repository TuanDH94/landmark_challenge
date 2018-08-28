import numpy as np
import librosa
import math
import re
import os
import struct
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
# import IPython.display as ipd

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
def wav_plotter(full_path, class_label):
    rate, wav_sample = wav.read(full_path)
    wave_file = open(full_path,"rb")
    riff_fmt = wave_file.read(36)
    bit_depth_string = riff_fmt[-2:]
    bit_depth = struct.unpack("H",bit_depth_string)[0]
    print('sampling rate: ',rate,'Hz')
    print('bit depth: ',bit_depth)
    #print('number of channels: ',wav_sample.shape[1])
    print('duration: ',wav_sample.shape[0]/rate,' second')
    print('number of samples: ',len(wav_sample))
    print('class: ',class_label)
    plt.figure(figsize=(12, 4))
    plt.plot(wav_sample)
    plt.show()
    #return ipd.Audio(full_path)
if __name__ == '__main__':
    file = 'E:\\Data\\train_voice\\train\\female_central\\22da883214754878af2101120bbfb2ee__lWbAHVYAXI_160-181_0.wav'
    data = getfeature(file)
    #wav_plotter(file, 'male')
    print('')