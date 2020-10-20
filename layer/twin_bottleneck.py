from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from layer import gcn
from layer.functional import build_adjacency


# noinspection PyAbstractClass
class TwinBottleneck(tf.keras.layers.Layer):
    def __init__(self, bbn_dim, cbn_dim, **kwargs):
        super().__init__(**kwargs)
        self.bbn_dim = bbn_dim
        self.cbn_dim = cbn_dim
        self.gcn = gcn.GCNLayer(cbn_dim)

    # noinspection PyMethodOverriding
    def call(self, bbn, cbn, context, step=-1):
        adj = build_adjacency(bbn)
        if step >= 0:
            sim = tf.expand_dims(tf.expand_dims(adj, 0), -1)
            tf.summary.image('vq/adj', sim, step=step, max_outputs=1)
            tf.summary.histogram('vq/hist', adj, step=step)
        return tf.nn.relu(self.gcn(cbn, adj)), adj


class MemoryBottleneck(tf.keras.layers.Layer):
    # noinspection PyMethodOverriding
    def call(self, x, memory, training=True, **kwargs):
        """

        :param x: [N D]
        :param memory: [M D]
        :param training:
        :param kwargs:
        :return:
        """

        _x = tf.squeeze(x)
        att = tf.matmul(_x, memory, transpose_b=True)
        d_model = tf.cast(tf.shape(memory)[1], dtype=tf.float32)
        att = tf.nn.softmax(att / tf.math.sqrt(d_model), axis=-1)  # [N M]

        y = tf.matmul(att, memory, transpose_b=True)

        return tf.expand_dims(y, 0)
