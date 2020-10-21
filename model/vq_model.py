from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from model.attentional_model import AttentionalModel
from layer.gcn import GCNLayer
from layer import functional
from util.data.basic_data import BasicData as Data
from util.data.processing import DATASET_EXAMPLE_COUNT, DATASET_SHAPE, DATASET_CLASS_COUNT, img_processing
from util import eval


class VQModel(AttentionalModel):

    def _agg(self):
        self.aggregator = GCNLayer(self.latent_size)
        self.fc = tf.keras.layers.Dense(self.latent_size)

    def call(self, inputs, training=None, mask=None, step=-1):

        if training:
            s1, s2, u1, u2, _, _ = inputs
            img = tf.concat([s1, u1], axis=0)
            _img = tf.concat([s2, u2], axis=0)

            feat = self.encoder(img, training=training)
            _feat = self.encoder(_img, training=training)

            cls_input = feat if self.share_encoder else img
            _cls_input = _feat if self.share_encoder else _img

            pre_vq = self.fc(feat)  # [N D]

            _, context_ind = functional.nearest_context(pre_vq, self.context)  # _ [N]
            af_vq = functional.vq(pre_vq, self.context)  # [N D]
            adj = functional.build_adjacency_v1(af_vq)  # [N N]

            bn = self.aggregator(feat, adj, training=training)

            recon = self.decoder(bn, training=training)
            pred = self.classifier(cls_input, self.context, training=training, step=step)
            _pred = self.classifier(_cls_input, self.context, training=training, step=step)

            return recon, pred, _pred, pre_vq, context_ind
        else:
            cls_input = self.encoder(inputs[0], training=training) if self.share_encoder else inputs[0]
            pred = self.classifier(cls_input, self.context, training=training)
            return pred

    # noinspection PyMethodOverriding
    def obj(self, original, label, recon, pred, _pred, pre_vq, context_ind, epoch, step=-1):

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

        indexed_emb = context_ind @ self.context
        kl_1 = tf.reduce_mean(tf.square(tf.stop_gradient(pre_vq) - indexed_emb)) / 2.
        kl_2 = .25 * tf.reduce_mean(tf.square(tf.stop_gradient(indexed_emb) - pre_vq)) / 2.

        loss = 2. * loss_ae + .5 * loss_cls + loss_cons * .5 + kl_1 + kl_2

        if step >= 0:
            tf.summary.scalar('loss_all/loss', loss, step=step)
            tf.summary.scalar('loss_vq/likelihood', loss_ae, step=step)
            tf.summary.scalar('loss_vq/cons', loss_cons, step=step)
            tf.summary.scalar('loss_vq/kl_1', kl_1, step=step)
            tf.summary.scalar('loss_vq/kl_2', kl_2, step=step)

        return loss


def step_train(model: VQModel, data: Data, opt: tf.keras.optimizers.Optimizer, step):
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
        x, pred, _pred, pre_vq, context_ind = model(batch_feed, training=True, step=summary_step)
        img = tf.concat([s1, u1], axis=0)
        loss = model.obj(img, s_d[1], x, pred, _pred, pre_vq, context_ind, epoch, step=summary_step)
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
