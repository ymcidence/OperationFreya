from __future__ import absolute_import, print_function, division, unicode_literals
from layer.cnn import _encoding_v1, CNNDecoder
import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.filter_size = (64, 128, 128)
        self.net = _encoding_v1(self.filter_size)

    def call(self, inputs, training=True, **kwargs):
        x = self.net(inputs, training=training)
        return tf.reduce_mean(x, axis=[1, 2])


Decoder = CNNDecoder
