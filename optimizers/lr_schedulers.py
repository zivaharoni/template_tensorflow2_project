import numpy as np
import tensorflow as tf
import logging

logger = logging.getLogger("logger")

class HalfCycleLrScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def get_config(self):
        pass

    def __init__(self, lr, max_steps, **kwargs):
        super().__init__(**kwargs)
        self.max_steps = max_steps
        self.lr = lr

    def __call__(self, step):
        return self.lr * (1 + tf.math.sin(tf.constant(np.pi, dtype=tf.float64) * step / (self.max_steps * 0.75)))

