from voice_classification.extract_feature import processing, multi_processing
from keras.layers.recurrent import RNN, LSTMCell, GRUCell, LSTM
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop
folder_train = 'E:\\Data\\train_voice\\debug\\'
nb_epochs =100
batch_size = 1
def get_model(time_series, nfeatures, nclass):
    model = Sequential()
    model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True,
                   input_shape=(time_series, nfeatures)))
    model.add(LSTM(units=64, dropout=0.05, recurrent_dropout=0.35, return_sequences=True))
    model.add(LSTM(units=16, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
    model.add(Dense(units=nclass, activation='softmax'))

    return model

def get_model_2(time_series, nfeatures, nclass):
    return
def load_data(case):
    if case == 1:
        X, Ygender, _ = multi_processing(folder_train)
        return train_test_split(X, Ygender)
    else:
        X, _, Yregion = multi_processing(folder_train)
        return train_test_split(X, Yregion)


if __name__ == '__main__':
    x_train, x_valid, y_train, y_valid = load_data(1)
    time_series = x_train.shape[1]
    nfeatures = x_train.shape[2]
    nclasses = 2
    model = get_model(time_series, nfeatures, nclasses)
    opt = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs, validation_data=(x_valid, y_valid), callbacks=[])

