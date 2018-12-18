from keras import Sequential
from keras.layers import Conv2D, Dropout, Flatten, Dense
from keras.layers import Lambda
from keras.regularizers import l2

input_width = 200
input_height = 66

def get_model():

    input_size = (input_height, input_width, 1)

    dense_keep_prob = 0.8
    init = 'glorot_uniform'

    model = Sequential()

    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_size))

    model.add(Conv2D(24, kernel_size=5, activation='relu', strides=(2, 2), kernel_initializer=init, kernel_regularizer=l2(0.001)))
    model.add(Conv2D(36, kernel_size=5, activation='relu', strides=(2, 2), kernel_initializer=init, kernel_regularizer=l2(0.001)))
    model.add(Conv2D(48, kernel_size=5, activation='relu', strides=(2, 2), kernel_initializer=init, kernel_regularizer=l2(0.001)))
    model.add(Conv2D(64, kernel_size=3, activation='relu', strides=(1, 1), kernel_initializer=init, kernel_regularizer=l2(0.001)))
    model.add(Conv2D(64, kernel_size=3, activation='relu', strides=(1, 1), kernel_initializer=init, kernel_regularizer=l2(0.001)))
    model.add(Flatten())
    model.add(Dense(units=1164, kernel_regularizer=l2(0.001)))
    model.add(Dropout(rate=dense_keep_prob))

    model.add(Dense(units=100, kernel_regularizer=l2(0.001)))
    model.add(Dense(units=50, kernel_regularizer=l2(0.001)))
    model.add(Dense(units=10, kernel_regularizer=l2(0.001)))
    model.add(Dense(units=1))

    return model
