import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Input

from typing import Tuple
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

    def train(self, inp: np.array, out: np.array, val_data: Tuple[np.array, np.array]) -> tf.keras.callbacks.History:
        return self.__model.fit(
            x=inp,
            y=out,
            batch_size=Const.batch_size,
            epochs=Const.epochs,
            verbose=2,
            validation_data=val_data
        )

    def test(self, inp: np.array, out: np.array) -> np.float32:
        return self.__model.evaluate(x=inp,
                                     y=out,
                                     verbose=2)

    def forecast(self, inp: np.array) -> np.array:
        return self.__model.predict(x=inp,
                                    verbose=2)
