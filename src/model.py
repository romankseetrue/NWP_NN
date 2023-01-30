import tensorflow as tf


class Model:
    def __init__(self) -> None:
        # Set random seed for reproducible results
        tf.keras.utils.set_random_seed(1)
        tf.config.experimental.enable_op_determinism()
