from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf


@tf.function
def row_distance_cosine(tensor_a, tensor_b):
    norm_a = tf.sqrt(tf.reduce_sum(tf.pow(tensor_a, 2), 1, keepdims=True))  # [N, 1]
    norm_b = tf.sqrt(tf.reduce_sum(tf.pow(tensor_b, 2), 1, keepdims=True))
    denominator = tf.matmul(norm_a, norm_b, transpose_b=True)
    numerator = tf.matmul(tensor_a, tensor_b, transpose_b=True)

    return numerator / denominator


def nearest_context(feature, context):
    distances = row_distance_cosine(feature, context) * -1
    min_ind = tf.cast(tf.argmin(distances, axis=1), dtype=tf.int32)
    k = tf.shape(context)[0]
    min_ind = tf.one_hot(min_ind, k, dtype=tf.float32)  # [N k]
    rslt = min_ind @ context
    return rslt, tf.stop_gradient(min_ind)


@tf.custom_gradient
def vq(feature, context):
    value, ind = nearest_context(feature, context)

    def grad(d_value):
        d_context = tf.matmul(ind, d_value, transpose_a=True)
        return d_value, d_context

    return value, grad


@tf.function
def build_adjacency(feature):
    """

    :param feature: [N d]
    :return:
    """
    adj = tf.nn.relu(row_distance_cosine(feature, feature))
    adj = tf.pow(adj, 1)
    return adj
