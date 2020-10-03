from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from layer import twin_bottleneck
from layer import functional


class AutoEncoding(tf.keras.layers.Layer):
    def __init__(self, encoder: tf.keras.layers.Layer, decoder: tf.keras.layers.Layer, latent_size, cls_num, l2=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.l2 = l2
        self.fc_1 = tf.keras.layers.Dense(latent_size)
        self.fc_2 = tf.keras.layers.Dense(latent_size)
        self.tbn = twin_bottleneck.TwinBottleneck(latent_size, latent_size)
        self.context = tf.Variable(initial_value=tf.random.normal([cls_num, latent_size], stddev=.01), trainable=True,
                                   dtype=tf.float32, name='ContextEmb')
        if l2:
            self.context = tf.nn.l2_normalize(self.context)

    def call(self, inputs, training=True, step=-1, **kwargs):
        feat = self.encoder(inputs, training=training)
        bbn = self.fc_1(feat, training=training)
        if self.l2:
            bbn = tf.nn.l2_normalize(bbn)
        _, context_ind = functional.nearest_context(bbn, self.context)
        vq_bbn = functional.vq(bbn, self.context)

        cbn = self.fc_2(feat, training=training)
        latent, adj = self.tbn(vq_bbn, cbn, self.context, step=step)
        return self.decoder(latent, training=training), bbn, context_ind, feat, adj

    def obj(self, original, decoded, bbn, context_ind, beta=.25, step=-1):
        """

        :param original: original images
        :param decoded: decoded images
        :param bbn: encoded bottleneck before vector quantization
        :param context_ind: the selected entries of each bottleneck feature during vector quantization
        :param beta: vq kld bp scaler
        :param step: for summary
        :return:
        """
        likelihood = tf.reduce_mean(tf.square(original - decoded)) / 2.

        indexed_emb = context_ind @ self.context

        kl_1 = tf.reduce_mean(tf.square(tf.stop_gradient(bbn) - indexed_emb)) / 2.
        kl_2 = beta * tf.reduce_mean(tf.square(tf.stop_gradient(indexed_emb) - bbn)) / 2.

        loss = likelihood + kl_1 + kl_2

        if step >= 0:
            sim = (functional.row_distance_cosine(self.context, self.context) + 1) / 2
            sim = tf.expand_dims(tf.expand_dims(sim, 0), -1)
            tf.summary.image('vq/emb', sim, step=step, max_outputs=1)
            tf.summary.scalar('loss_vq/likelihood', likelihood, step=step)
            tf.summary.scalar('loss_vq/kl_1', kl_1, step=step)
            tf.summary.scalar('loss_vq/kl_2', kl_2, step=step)
            tf.summary.scalar('loss_vq/loss', loss, step=step)
        return loss
