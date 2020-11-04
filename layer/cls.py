from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from layer.functional import row_distance


class DistanceClassifier(tf.keras.layers.Layer):
    def __init__(self, latent_size, class_num, backbone=None, temp=.07, l2=False, **kwargs):
        """

        :param backbone: a backbone encoder
        :param latent_size: emb size
        :param class_num:
        :param temp: classification temperature
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.backbone = backbone
        self.latent_size = latent_size
        self.l2 = l2
        self.temp = temp
        self.class_num = class_num

    # noinspection PyMethodOverriding
    def call(self, inputs, context, training=True, step=-1, **kwargs) -> tf.Tensor:
        x = self.backbone(inputs, training=training) if self.backbone is not None else inputs
        distances = -1 * row_distance(x, context) / self.temp
        return distances

    @staticmethod
    def obj(label, pred, step=-1):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(label), logits=pred))
        if step >= 0:
            tf.summary.scalar('loss_cls/cls', loss, step=step)

        return loss

class ContextClassifier(tf.keras.layers.Layer):
    def __init__(self, latent_size, class_num, backbone=None, temp=.07, l2=False, **kwargs):
        """
        
        :param backbone: a backbone encoder
        :param latent_size: emb size
        :param class_num:
        :param temp: classification temperature
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.backbone = backbone
        self.latent_size = latent_size
        self.l2 = l2
        self.bias = tf.Variable(initial_value=tf.zeros([class_num], dtype=tf.float32), trainable=True,
                                dtype=tf.float32, name='bias')
        self.temp = temp
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(latent_size),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(.3)
        ])
        # self.bn = tf.keras.layers.BatchNormalization()

    # noinspection PyMethodOverriding
    def call(self, inputs, context, training=True, step=-1, **kwargs) -> tf.Tensor:
        """

        :param inputs: [N ...]
        :param context: [M d]
        :param training:
        :param step:
        :param kwargs:
        :return:
        """
        x = self.backbone(inputs, training=training) if self.backbone is not None else inputs
        x = self.fc(x, training=training)
        if self.l2:
            x = tf.nn.l2_normalize(x, axis=1)  # [N d]
        x = tf.matmul(x, context, transpose_b=True)  # + self.bias
        x = x / self.temp
        return x

    @staticmethod
    def obj(label, pred, step=-1):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(label), logits=pred))
        if step >= 0:
            tf.summary.scalar('loss_cls/cls', loss, step=step)

        return loss
