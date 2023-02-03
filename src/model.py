import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Input

from typing import List
from const import Const
from prepare_data import Query, TrainValDataLoader


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
        self.__input_data_shape = (-1, Const.measurements_per_day, 1)

    def train(self, train_data_queries: List[Query], val_data_queries: List[Query]) -> tf.keras.callbacks.History:
        train_observations: np.array
        train_forecasts: np.array
        train_observations, train_forecasts = TrainValDataLoader(
            train_data_queries).get_data()
        train_forecasts = train_forecasts.reshape(self.__input_data_shape)

        val_observations: np.array
        val_forecasts: np.array
        val_observations, val_forecasts = TrainValDataLoader(
            val_data_queries).get_data()
        val_forecasts = val_forecasts.reshape(self.__input_data_shape)

        return self.__model.fit(
            x=train_forecasts,
            y=train_observations,
            batch_size=Const.batch_size,
            epochs=Const.epochs,
            verbose=Const.verbose,
            validation_data=(val_forecasts, val_observations)
        )

    def test(self, inp: np.array, out: np.array) -> np.float32:
        inp = inp.reshape(self.__input_data_shape)
        return self.__model.evaluate(x=inp,
                                     y=out,
                                     verbose=Const.verbose)

    def forecast(self, inp: np.array) -> np.array:
        inp = inp.reshape(self.__input_data_shape)
        return self.__model.predict(x=inp,
                                    verbose=Const.verbose)
