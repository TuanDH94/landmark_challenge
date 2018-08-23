import keras
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

data = np.array([[i, j] for i in range(50) for j in range(100)])
targets = np.array([[i] for i in range(50)])

data_gen = TimeseriesGenerator(data, targets,
                               length=10, sampling_rate=1,
                               batch_size=2)
pad = keras.preprocessing.sequence.pad_sequences(data, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0)

for x, y in data_gen:
    print('')
print('')