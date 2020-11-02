from __future__ import absolute_import, print_function, division, unicode_literals
from layer.cnn import _encoding_v1, CNNDecoder, _encoding_v2
import tensorflow as tf


class EncoderV1(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.filter_size = (96, 192, 192)
        self.net = _encoding_v1(self.filter_size)

    def call(self, inputs, training=True, **kwargs):
        x = self.net(inputs, training=training)
        return tf.reduce_mean(x, axis=[1, 2])


class EncoderV2(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = _encoding_v2()

    def call(self, inputs, training=True, **kwargs):
        x = self.net(inputs, training=training)
        return tf.reduce_mean(x, axis=[1, 2])


Encoder = EncoderV2
Decoder = CNNDecoder
