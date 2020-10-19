from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_addons as tfa
from layer.functional import Dummy


def _encoding_v1(filter_size=(64, 128, 128), p=.5):
    def _conv_unit(_f, _k):
        return tf.keras.Sequential([tf.keras.layers.Conv2D(_f, _k, padding='SAME', data_format='channels_last'),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.LeakyReLU(.1)])

    net = tf.keras.Sequential([_conv_unit(filter_size[0], 3),
                               _conv_unit(filter_size[0], 3),
                               _conv_unit(filter_size[0], 3),
                               tf.keras.layers.MaxPool2D(strides=2, padding='SAME', data_format='channels_last'),
                               tf.keras.layers.Dropout(p),
                               _conv_unit(filter_size[1], 3),
                               _conv_unit(filter_size[1], 3),
                               _conv_unit(filter_size[1], 3),
                               tf.keras.layers.MaxPool2D(strides=2, padding='SAME', data_format='channels_last'),
                               tf.keras.layers.Dropout(p),
                               _conv_unit(filter_size[2], 3),
                               _conv_unit(filter_size[2], 1),
                               _conv_unit(filter_size[2], 1)])

    return net


def _decoding_v1():
    def _t_conv_unit(_f, _k, _s, _p):
        return tf.keras.Sequential(
            [tf.keras.layers.Conv2DTranspose(_f, _k, _s, padding='SAME', output_padding=_p, data_format='channel_last'),
             tf.keras.layers.BatchNormalization(),
             tf.keras.layers.ReLU()])

    net = tf.keras.Sequential([
        tf.keras.layers.Dense(4 * 4 * 512),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Reshape((4, 4, 512)),
        _t_conv_unit(256, 5, 2, 1),
        _t_conv_unit(128, 5, 2, 1),
        tfa.layers.WeightNormalization(
            tf.keras.layers.Conv2DTranspose(3, 5, 2, padding='SAME', output_padding=1, data_format='channel_last')),
        Dummy(tf.nn.tanh)])

    return net


class SVHNEncoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.filter_size = (64, 128, 128)
        self.net = _encoding_v1(self.filter_size)

    def call(self, inputs, training=True, **kwargs):
        x = self.net(inputs)
        return tf.reduce_mean(x, axis=[1, 2])


class CIFAREncoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.filter_size = (96, 192, 192)
        self.net = _encoding_v1(self.filter_size)

    def call(self, inputs, training=True, **kwargs):
        x = self.net(inputs, training=training)
        return tf.reduce_mean(x, axis=[1, 2])


class SVHNDecoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = _decoding_v1()

    def call(self, inputs, training=True, **kwargs):
        return self.net(inputs, training=training)


Encoder = SVHNEncoder
Decoder = SVHNDecoder
