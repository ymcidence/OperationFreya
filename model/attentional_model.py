from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np

from layer import mnist, cls, cifar, svhn
from layer.transformer.isab import InducedSetAttentionBlock
from layer.twin_bottleneck import MemoryBottleneck
from util.data.basic_data import BasicData as Data
from util.data.processing import DATASET_CLASS_COUNT, DATASET_SHAPE, DATASET_EXAMPLE_COUNT, img_processing
from util import eval

ENCODEC = {'mnist': mnist,
           'cifar10': cifar,
           'svhn': svhn,
           'svhn_extra': svhn,
           'cifar_unnormalized': cifar}


class AttentionalModel(tf.keras.Model):
    def __init__(self, set_name, latent_size, class_num, share_encoder=True, temp=1., l2=False,
                 para_cls=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_name = set_name
        self.latent_size = latent_size
        self.class_num = class_num
        self.share_encoder = share_encoder
        self.temp = temp
        self.l2 = l2
        self.context = tf.Variable(initial_value=tf.random.normal([class_num, latent_size], stddev=.01), trainable=True,
                                   dtype=tf.float32, name='ContextEmb')

        self.encodec = ENCODEC[set_name]

        self.encoder = self.encodec.Encoder()
        self.decoder = self.encodec.Decoder()
        self._agg()
        cls_backbone = None if share_encoder else self.encodec.Encoder()
        self.classifier = cls.ContextClassifier(latent_size, class_num, cls_backbone, temp,
                                                l2=l2) if para_cls else cls.DistanceClassifier(latent_size, class_num,
                                                                                               cls_backbone, temp,
                                                                                               l2=l2)

    def _agg(self):
        self.aggregator = InducedSetAttentionBlock(self.latent_size, 1, self.latent_size)

    def call(self, inputs, training=None, mask=None, step=-1):

        if training:
            s1, s2, u1, u2, _, _ = inputs
            img = tf.concat([s1, u1], axis=0)
            _img = tf.concat([s2, u2], axis=0)

            feat = self.encoder(img, training=training)
            _feat = self.encoder(_img, training=training)

            cls_input = feat if self.share_encoder else img
            _cls_input = _feat if self.share_encoder else _img

            # Original transformers accept [n t d] tensors. We have to reshape the encoded features into a [1 n d] tensor
            feat = tf.expand_dims(feat, axis=0)
            bn = self.aggregator(feat, self.context, training=training)
            bn = tf.squeeze(bn)

            recon = self.decoder(bn, training=training)
            pred = self.classifier(cls_input, self.context, training=training, step=step)
            _pred = self.classifier(_cls_input, self.context, training=training, step=step)

            return recon, pred, _pred
        else:
            cls_input = self.encoder(inputs[0], training=training) if self.share_encoder else inputs[0]
            pred = self.classifier(cls_input, self.context, training=training)
            return pred

    def obj(self, original, label, recon, pred, _pred, epoch, step=-1):

        loss_ae = tf.reduce_mean(tf.square(tf.stop_gradient(original) - recon)) / 2.
        if tf.shape(label).shape[0] < 2:
            ll = tf.one_hot(label, self.class_num)
        else:
            ll = label
        s_size = tf.shape(label)[0]
        s_pred = pred[:s_size, :]
        loss_cls = self.classifier.obj(ll, s_pred, step)

        softmax_pred = tf.nn.softmax(pred)
        _softmax_pred = tf.nn.softmax(_pred)

        loss_cons = tf.reduce_mean(tf.square(tf.stop_gradient(softmax_pred) - _softmax_pred)) / 2.

        ramp = np.exp(-5 * (1 - epoch) * (1 - epoch))

        loss = loss_ae + .5 * loss_cls + loss_cons * .5

        if step >= 0:
            tf.summary.scalar('loss_all/loss', loss, step=step)
            tf.summary.scalar('loss_vq/likelihood', loss_ae, step=step)
            tf.summary.scalar('loss_vq/cons', loss_cons, step=step)
            tf.summary.scalar('loss_vq/ramp', ramp, step=step)

        return loss


class MBModel(AttentionalModel):
    def _agg(self):
        self.aggregator = MemoryBottleneck()


def step_train(model: AttentionalModel, data: Data, opt: tf.keras.optimizers.Optimizer, step):
    epoch = step // (DATASET_EXAMPLE_COUNT['train'][data.set_name] / data.batch_size)
    s_d, u_d = data.next_train()

    s1 = img_processing(s_d[0])
    s2 = img_processing(s_d[0])
    u1 = img_processing(u_d[0])
    u2 = img_processing(u_d[0])
    if data.set_name == 'mnist':
        shape = np.prod(DATASET_SHAPE[data.set_name][1:])
        s1 = tf.reshape(img_processing(s_d[0]), [-1, shape])
        s2 = tf.reshape(img_processing(s_d[0]), [-1, shape])
        u1 = tf.reshape(img_processing(u_d[0]), [-1, shape])
        u2 = tf.reshape(img_processing(u_d[0]), [-1, shape])
    batch_feed = [s1, s2, u1, u2, s_d[1], u_d[1]]
    batch_feed = [tf.identity(i) for i in batch_feed]
    summary_step = -1 if step % 50 > 0 else step
    with tf.GradientTape() as tape:
        x, pred, _pred = model(batch_feed, training=True, step=summary_step)
        img = tf.concat([s1, u1], axis=0)
        loss = model.obj(img, s_d[1], x, pred, _pred, epoch, step=summary_step)
        gradient = tape.gradient(loss, sources=model.trainable_variables)
        opt.apply_gradients(zip(gradient, model.trainable_variables))
    if summary_step >= 0:
        acc = eval.acc(s_d[1], pred[:data.s_size, :])
        err = 1 - acc

        ll = tf.concat([batch_feed[2], batch_feed[3]], axis=0)
        if tf.shape(ll).shape[0] < 2:
            ll = tf.one_hot(ll, DATASET_CLASS_COUNT[data.set_name])

        sim = tf.expand_dims(tf.expand_dims(tf.matmul(ll, ll, transpose_b=True), 0), -1)
        # tf.summary.image('img', s_d[0], step=step, max_outputs=1)
        # tf.summary.image('gt/adj', sim, step=step, max_outputs=1)
        tf.summary.scalar('train/acc', acc, step=step)
        tf.summary.scalar('train/err', err, step=step)

    return loss


def step_val(model: AttentionalModel, data: Data, step):
    t_d = data.next_test()
    d = t_d[0]
    if data.set_name == 'mnist':
        shape = np.prod(DATASET_SHAPE[data.set_name][1:])
        d = tf.reshape(t_d[0], [-1, shape])
    pred = model([d], training=False)

    acc = eval.acc(t_d[1], pred)
    err = 1 - acc
    tf.summary.scalar('val/acc', acc, step=step)
    tf.summary.scalar('val/err', err, step=step)

    return acc
