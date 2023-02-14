import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Input

from typing import List, Optional
from const import Const
from prepare_data import Query, TrainValDataLoader, Sampler


class Model:
    def __init__(self) -> None:
        # Set random seed for reproducible results
        tf.keras.utils.set_random_seed(1)
        tf.config.experimental.enable_op_determinism()

        self.__model: Optional[keras.Model] = None

    def train(self, train_data_queries: List[Query], val_data_queries: List[Query], samples_designer: Sampler) -> tf.keras.callbacks.History:
        train_forecasts: np.array
        train_observations: np.array
        train_forecasts, train_observations = TrainValDataLoader(
            train_data_queries, samples_designer).get_data()

        val_forecasts: np.array
        val_observations: np.array
        val_forecasts, val_observations = TrainValDataLoader(
            val_data_queries, samples_designer).get_data()

        return self.__model.fit(
            x=train_forecasts,
            y=train_observations,
            batch_size=Const.batch_size,
            epochs=Const.epochs,
            verbose=Const.verbose,
            validation_data=(val_forecasts, val_observations)
        )

    def test(self, inp: np.array, out: np.array) -> np.float32:
        return self.__model.evaluate(x=inp,
                                     y=out,
                                     verbose=Const.verbose)

    def forecast(self, inp: np.array) -> np.array:
        return self.__model.predict(x=inp,
                                    verbose=Const.verbose)


class CosmoModel(Model):
    def __init__(self) -> None:
        super().__init__()

        self._Model__model: keras.Model = Sequential()
        self._Model__model.add(Input(shape=(None, 1)))
        self._Model__model.add(GRU(16))
        self._Model__model.add(Dense(Const.measurements_per_day))

        self._Model__model.summary()

        self._Model__model.compile(loss='mse', optimizer='adam')


class ClimateModel(Model):
    def __init__(self) -> None:
        super().__init__()

        self._Model__model: keras.Model = Sequential()
        self._Model__model.add(Input(shape=(None, 1)))
        self._Model__model.add(GRU(16, return_sequences=True))
        self._Model__model.add(GRU(16))
        self._Model__model.add(Dense(1))

        self._Model__model.summary()

        self._Model__model.compile(loss='mse', optimizer='adam')
