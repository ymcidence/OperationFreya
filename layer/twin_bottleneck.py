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
