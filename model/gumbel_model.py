from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from model.attentional_model import ENCODEC
from layer import functional
from layer.gcn import GCNLayer
from util.data.basic_data import BasicData as Data
from util.data.processing import DATASET_EXAMPLE_COUNT, DATASET_SHAPE, img_processing
from util import eval


def sample_gumbel(shape, eps=1e-10):
    u = tf.random.uniform(shape, minval=0, maxval=1)
    return -tf.math.log(-tf.math.log(u + eps) + eps)


def gumbel_softmax(logits, temperature, hard=False):
    gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
    y = tf.nn.softmax(gumbel_softmax_sample / temperature)

    if hard:
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)),
                         y.dtype)
        y = tf.stop_gradient(y_hard - y) + y

    return y


class GumbelModel(tf.keras.Model):
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
        self.bias = tf.Variable(initial_value=tf.zeros([class_num], dtype=tf.float32), trainable=True,
                                dtype=tf.float32, name='bias')

        self.encodec = ENCODEC[set_name]
        self.fc1 = tf.keras.Sequential([
            tf.keras.layers.Dense(self.latent_size),
            tf.keras.layers.BatchNormalization()
        ])

        self.encoder = self.encodec.Encoder()
        self.decoder = self.encodec.Decoder()
        self._agg()

    def _agg(self):
        self.aggregator = GCNLayer(self.latent_size)

    def call(self, inputs, training=None, mask=None, step=-1):
        if training:
            s1, s2, u1, u2, _, _ = inputs
            img = tf.concat([s1, u1], axis=0)
            _img = tf.concat([s2, u2], axis=0)
            enc = self.encoder(img, training=training)
            _enc = self.encoder(_img, training=training)

            feat = self.fc1(enc, training=training)
            _feat = self.fc1(_enc, training=training)

            cls_input = feat if self.share_encoder else img
            _cls_input = _feat if self.share_encoder else _img

            logits = tf.matmul(cls_input, self.context, transpose_b=True) + self.bias
            _logits = tf.matmul(_cls_input, self.context, transpose_b=True) + self.bias

            gumbel_prob = gumbel_softmax(logits, .1)
            gumbel_feat = gumbel_prob @ self.context

            adj = functional.build_adjacency_v1(gumbel_feat)
            bn = self.aggregator(enc, adj, training=training)

            recon = self.decoder(bn, training=training)

            return recon, logits, _logits
        else:
            cls_input = self.encoder(inputs[0], training=training) if self.share_encoder else inputs[0]
            return tf.matmul(cls_input, self.context, transpose_b=True) + self.bias

    def obj(self, original, label, recon, pred, _pred, epoch, step=-1):

        loss_ae = tf.reduce_mean(tf.square(tf.stop_gradient(original) - recon)) / 2.
        if tf.shape(label).shape[0] < 2:
            ll = tf.one_hot(label, self.class_num)
        else:
            ll = label
        s_size = tf.shape(label)[0]
        s_pred = pred[:s_size, :]
        loss_cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(ll), logits=s_pred))

        softmax_pred = tf.nn.softmax(pred)
        _softmax_pred = tf.nn.softmax(_pred)

        loss_cons = tf.reduce_mean(tf.square(tf.stop_gradient(softmax_pred) - _softmax_pred)) / 2.

        ramp = np.exp(-5 * np.sqrt((1 - epoch) * (1 - epoch)))

        loss = loss_ae + loss_cls + loss_cons * ramp

        if step >= 0:
            tf.summary.scalar('loss_all/loss', loss, step=step)
            tf.summary.scalar('loss_vq/likelihood', loss_ae, step=step)
            tf.summary.scalar('loss_vq/cons', loss_cons, step=step)
            tf.summary.scalar('loss_vq/ramp', ramp, step=step)
            tf.summary.scalar('loss_cls/cls', loss_cls, step=step)

        return loss, loss_cls + loss_cons * ramp


def step_train(model: GumbelModel, data: Data, opt1: tf.keras.optimizers.Optimizer, opt2: tf.keras.optimizers.Optimizer,
               step):
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
    with tf.GradientTape() as tape1:  # , tf.GradientTape() as tape2:
        x, pred, _pred = model(batch_feed, training=True, step=summary_step)
        img = tf.concat([s1, u1], axis=0)
        loss1, loss2 = model.obj(img, s_d[1], x, pred, _pred, epoch, step=summary_step)
        gradient1 = tape1.gradient(loss1 + loss2, sources=model.trainable_variables)
        # gradient2 = tape2.gradient(loss2, sources=model.trainable_variables)
        opt1.apply_gradients(zip(gradient1, model.trainable_variables))
        # opt2.apply_gradients(zip(gradient2, model.trainable_variables))
    if summary_step >= 0:
        acc = eval.acc(s_d[1], pred[:data.s_size, :])
        err = 1 - acc

        # tf.summary.image('img', s_d[0], step=step, max_outputs=1)
        # tf.summary.image('gt/adj', sim, step=step, max_outputs=1)
        tf.summary.scalar('train/acc', acc, step=step)
        tf.summary.scalar('train/err', err, step=step)

    return loss1 + loss2


def step_val(model: GumbelModel, data: Data, step):
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
