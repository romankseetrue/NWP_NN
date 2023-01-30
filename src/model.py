import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Input

from const import Const


def create_model() -> keras.Model:
    model: keras.Model = Sequential()
    model.add(Input(shape=(None, 1)))
    model.add(GRU(16))
    model.add(Dense(Const.measurements_per_day))

    model.summary()

    return model


class Model:
    def __init__(self) -> None:
        # Set random seed for reproducible results
        tf.keras.utils.set_random_seed(1)
        tf.config.experimental.enable_op_determinism()

        self.__model: keras.Model = create_model()
        self.__model.compile(loss='mse', optimizer='adam')
