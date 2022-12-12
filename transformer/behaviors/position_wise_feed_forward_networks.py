from keras.layers import Dense
from keras.models import Sequential


def ffn(d_ff = 2048, d_model = 512, activation='relu'):
    return Sequential([
        Dense(units=d_ff, activation=activation),
        Dense(units=d_model)
    ])