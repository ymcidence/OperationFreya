from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os



def acc(label, pred):
    if tf.shape(label).shape[0] >= 2:
        label = tf.argmax(label, axis=-1)
    pred = tf.argmax(pred, axis=-1, output_type=tf.int32)
    compare = tf.cast(tf.equal(pred, label), tf.float32)

    return tf.reduce_mean(compare)


def err(label, pred):
    return 1 - acc(label, pred)

def get_tfevent_list(dir, file_list):
    newDir = dir

    if os.path.isfile(dir) and str(dir).find('tfevents') >= 0:
        file_list.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            get_tfevent_list(newDir, file_list)
    return file_list