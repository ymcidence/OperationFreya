from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


if __name__ == '__main__':
    step = tf.constant(1.)
    warmup_steps = 2000
    d_model = 256.

    for i in range(0, 10000, 100):
        step = tf.constant(i, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (warmup_steps ** -1.5)
        print(i, tf.math.rsqrt(d_model) * tf.math.minimum(arg1, arg2))


def ramp_up(e):
    if e <= 80:
        return np.exp(-5 * (1 - e / 80) * (1 - e / 80))
    elif e <= 250:
        return 1
    elif e <= 300:
        return np.exp(-12.5 * (1 - (300 - e) / 50) * (1 - (300 - e) / 50))
    else:
        return np.exp(-12.5 * (1 - (300 - 300) / 50) * (1 - (300 - 300) / 50))
