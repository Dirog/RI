import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
from keras import layers


def model(input_size, hidden_size, loss):
    input_signal = keras.Input(shape=(input_size,))

    layer1 = layers.Dense(input_size, activation='relu')(input_signal)
    layer2 = layers.Dense(64, activation='relu')(layer1)
    layer3 = layers.Dense(32, activation='relu')(layer2)
    encoded = layers.Dense(hidden_size, activation='relu')(layer3)
    layer5 = layers.Dense(32, activation='relu')(encoded)
    layer6 = layers.Dense(64, activation='relu')(layer5)
    decoded = layers.Dense(input_size, activation='relu')(layer6)

    autoencoder = keras.Model(input_signal, decoded)
    encoder = keras.Model(input_signal, encoded)

    autoencoder.compile(optimizer='adam', loss=loss)
    autoencoder.summary()

    return autoencoder, encoder