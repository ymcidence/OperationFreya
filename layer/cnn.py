from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from layer.functional import Dummy


def _encoding_v1(filter_size=(64, 128, 128), p=.5):
    def _conv_unit(_f, _k, padding='SAME'):
        return tf.keras.Sequential([tf.keras.layers.Conv2D(_f, _k, padding=padding, data_format='channels_last'),
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
                               _conv_unit(filter_size[2], 3, 'VALID'),
                               _conv_unit(filter_size[2], 1),
                               _conv_unit(filter_size[2], 1)])

    return net


def _encoding_v2():
    def _conv_unit(_f, _k, padding='SAME'):
        return tf.keras.Sequential([tf.keras.layers.Conv2D(_f, _k, padding=padding, data_format='channels_last'),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.LeakyReLU(.1)])

    net = tf.keras.Sequential([_conv_unit(128, 3),
                               _conv_unit(128, 3),
                               _conv_unit(128, 3),
                               tf.keras.layers.MaxPool2D(strides=2, data_format='channels_last'),
                               tf.keras.layers.Dropout(.5),
                               _conv_unit(256, 3),
                               _conv_unit(256, 3),
                               _conv_unit(256, 3),
                               tf.keras.layers.MaxPool2D(strides=2, data_format='channels_last'),
                               tf.keras.layers.Dropout(.5),
                               _conv_unit(512, 3, padding='VALID'),
                               _conv_unit(256, 3),
                               _conv_unit(128, 3)])
    return net


def _decoding_v1():
    def _t_conv_unit(_f, _k, _s, _p):
        return tf.keras.Sequential(
            [tf.keras.layers.Conv2DTranspose(_f, _k, _s, padding='SAME', output_padding=_p,
                                             data_format='channels_last'),
             tf.keras.layers.BatchNormalization(),
             tf.keras.layers.ReLU()])

    net = tf.keras.Sequential([
        tf.keras.layers.Dense(4 * 4 * 512),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Reshape((4, 4, 512)),
        _t_conv_unit(256, 5, 2, 1),
        _t_conv_unit(128, 5, 2, 1),
        tf.keras.layers.Conv2DTranspose(3, 5, 2, padding='SAME', output_padding=1, data_format='channels_last'),
        tf.keras.layers.BatchNormalization()])

    return net


class CNNDecoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = _decoding_v1()

    def call(self, inputs, training=True, **kwargs):
        return self.net(inputs, training=training)


if __name__ == '__main__':
    model = _encoding_v1()
    img = tf.zeros([5, 32, 32, 3])
    print(model(img))
