from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np

from layer import mnist, cls
from layer.transformer.isab import InducedSetAttentionBlock
from util.data.basic_data import BasicData as Data
from util.data.processing import DATASET_CLASS_COUNT, DATASET_SHAPE
from util import eval

ENCODEC = {'mnist': mnist}


class AttentionalModel(tf.keras.Model):
    def __init__(self, set_name, latent_size, class_num, share_encoder=True, temp=1., l2=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_name = set_name
        self.latent_size = latent_size
        self.class_num = class_num
        self.share_encoder = share_encoder
        self.temp = temp
        self.l2 = l2
        self.context = tf.Variable(initial_value=tf.random.normal([class_num, latent_size], stddev=.01), trainable=True,
                                   dtype=tf.float32, name='ContextEmb')

        self.encodec = ENCODEC.get(set_name, mnist)

        self.encoder = self.encodec.Encoder()
        self.decoder = self.encodec.Decoder()
        self.aggregator = InducedSetAttentionBlock(latent_size, 1, latent_size)

        cls_backbone = None if share_encoder else self.encodec.Encoder()
        self.classifier = cls.ContextClassifier(latent_size, class_num, cls_backbone, temp, l2=l2)

    def call(self, inputs, training=None, mask=None, step=-1):
        if training:
            img = tf.concat([inputs[0], inputs[1]], axis=0)
            s_size = tf.shape(inputs[0])[0]

            feat = self.encoder(img, training=training)
            cls_input = tf.stop_gradient(feat[:s_size, ...]) if self.share_encoder else img[:s_size, ...]

            # Original transformers accept [n t d] tensors. We have to reshape the encoded features into a [1 n d] tensor
            feat = tf.expand_dims(feat, axis=0)
            bn = self.aggregator(feat, self.context, training=training)
            bn = tf.squeeze(bn)

            recon = self.decoder(bn, training=training)
            pred = self.classifier(cls_input, self.context, training=training, step=step)

            return recon, pred
        else:
            cls_input = self.encoder(inputs[0], training=training) if self.share_encoder else inputs[0]
            pred = self.classifier(cls_input, self.context, training=training)
            return pred

    def obj(self, original, label, recon, pred, step=-1):
        loss_ae = tf.reduce_mean(tf.square(original - recon)) / 2.
        if tf.shape(label).shape[0] < 2:
            ll = tf.one_hot(label, self.class_num)
        else:
            ll = label
        loss_cls = self.classifier.obj(ll, pred, step)
        loss = loss_ae + .1 * loss_cls

        if step >= 0:
            tf.summary.scalar('loss_all/loss', loss, step=step)
            tf.summary.scalar('loss_vq/likelihood', loss_ae, step=step)

        return loss


def step_train(model: AttentionalModel, data: Data, opt: tf.keras.optimizers.Optimizer, step):
    s_d, u_d = data.next_train()
    shape = np.prod(DATASET_SHAPE[data.set_name][1:])
    batch_feed = [tf.reshape(s_d[0], [-1, shape]), tf.reshape(u_d[0], [-1, shape]), s_d[1], u_d[1]]
    batch_feed = [tf.identity(i) for i in batch_feed]
    summary_step = -1 if step % 50 > 0 else step
    with tf.GradientTape() as tape:
        x, pred = model(batch_feed, training=True, step=summary_step)
        img = tf.concat([batch_feed[0], batch_feed[1]], axis=0)
        loss = model.obj(img, s_d[1], x, pred, step=summary_step)
        gradient = tape.gradient(loss, sources=model.trainable_variables)
        opt.apply_gradients(zip(gradient, model.trainable_variables))
    if summary_step >= 0:
        acc = eval.acc(s_d[1], pred)
        err = 1 - acc

        ll = tf.concat([batch_feed[2], batch_feed[3]], axis=0)
        if tf.shape(ll).shape[0] < 2:
            ll = tf.one_hot(ll, DATASET_CLASS_COUNT[data.set_name])

        sim = tf.expand_dims(tf.expand_dims(tf.matmul(ll, ll, transpose_b=True), 0), -1)
        tf.summary.image('gt/adj', sim, step=step, max_outputs=1)
        tf.summary.scalar('train/acc', acc, step=step)
        tf.summary.scalar('train/err', err, step=step)

    return loss


def step_val(model: AttentionalModel, data: Data, step):
    t_d = data.next_test()
    shape = np.prod(DATASET_SHAPE[data.set_name][1:])
    d = tf.reshape(t_d[0], [-1, shape])
    pred = model([d], training=False)

    acc = eval.acc(t_d[1], pred)
    err = 1 - acc
    tf.summary.scalar('val/acc', acc, step=step)
    tf.summary.scalar('val/err', err, step=step)

    return acc
