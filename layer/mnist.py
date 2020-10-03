from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import tensorflow_addons as tfa


class Dummy(tf.keras.layers.Layer):
    def __init__(self, f, **kwargs):
        super().__init__(**kwargs)
        self.f = f

    def call(self, inputs, **kwargs):
        return self.f(inputs)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = tf.keras.Sequential([
            tf.keras.layers.GaussianNoise(stddev=.3),
            tf.keras.layers.Dense(1024, activation=tf.nn.relu),
            tf.keras.layers.GaussianNoise(stddev=.5),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.GaussianNoise(stddev=.5),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.GaussianNoise(stddev=.5),
            tf.keras.layers.Dense(256, activation=tf.nn.relu)
        ])

    def call(self, inputs, training=True, **kwargs):
        return self.net(inputs, training=training)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(512),
            tf.keras.layers.BatchNormalization(),
            Dummy(tf.nn.softplus),
            tf.keras.layers.Dense(512),
            tf.keras.layers.BatchNormalization(),
            Dummy(tf.nn.softplus),
            tfa.layers.WeightNormalization(tf.keras.layers.Dense(784, tf.nn.sigmoid))
        ])

    def call(self, inputs, training=True, **kwargs):
        return self.net(inputs, training=training)
