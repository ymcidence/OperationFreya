from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from layer import mnist, auto_encoding, cls

ENCODEC = {'mnist': mnist}


class BasicModel(tf.keras.Model):
    def __init__(self, set_name, latent_size, class_num, share_encoder=True, temp=1., l2=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_name = set_name
        self.latent_size = latent_size
        self.class_num = class_num
        self.share_encoder = share_encoder
        self.temp = temp
        self.l2 = l2
        self.encodec = ENCODEC.get(set_name, mnist)

        self.encoder = self.encodec.Encoder()
        self.decoder = self.encodec.Decoder()
        self.ae = auto_encoding.AutoEncoding(self.encoder, self.decoder, latent_size, class_num, l2=l2)

        cls_backbone = None if share_encoder else self.encodec.Encoder()
        self.classifier = cls.ContextClassifier(latent_size, class_num, cls_backbone, temp, l2=l2)

    def call(self, inputs, training=None, mask=None, step=-1):
        """

        :param inputs: a list containing 4 elements of [s_x, ux. s_l, u_l]
        :param training:
        :param mask:
        :param step:
        :return:
        """
        if training:
            img = tf.concat([inputs[0], inputs[1]], axis=0)

            x, bbn, context_ind, feat, adj = self.ae(img, training=training, step=step)

            s_size = tf.shape(inputs[0])[0]

            cls_input = feat[:s_size, ...] if self.share_encoder else img[:s_size, ...]
            pred = self.classifier(cls_input, self.ae.context, training=training, step=step)

            return x, bbn, context_ind, feat, pred, adj
        else:
            cls_input = self.ae(inputs[0], training=training)[3] if self.share_encoder else inputs[0]
            pred = self.classifier(cls_input, self.ae.context, training=training)
            return pred

    def obj(self, img, label, x, bbn, context_ind, feat, adj, pred, beta=.25, step=-1):
        loss_ae = self.ae.obj(img, x, bbn, context_ind, beta, step)

        if tf.shape(label).shape[0] < 2:
            ll = tf.one_hot(label, self.class_num)
        else:
            ll = label

        gt_adj = tf.matmul(ll, ll, transpose_b=True)
        s_size = tf.shape(ll)[0]
        sub_adj = adj[:s_size, :s_size]

        loss_adj = tf.reduce_mean(tf.square(sub_adj - gt_adj)) / 2.

        loss_cls = self.classifier.obj(ll, pred, step)

        loss = loss_ae + loss_cls + loss_adj

        if step >= 0:
            tf.summary.scalar('loss_all/loss', loss, step=step)
            tf.summary.scalar('loss_vq/loss_adj', loss_adj, step=step)

        return loss
