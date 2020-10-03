from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
from time import gmtime, strftime
from model.basic_model import BasicModel as Model
from util.data.basic_data import BasicData as Data
from util.data.processing import DATASET_CLASS_COUNT, DATASET_SHAPE
from util import eval  # , scheduler

from meta import ROOT_PATH


def step_train(model: Model, data: Data, opt: tf.keras.optimizers.Optimizer, step):
    s_d, u_d = data.next_train()
    shape = np.prod(DATASET_SHAPE[data.set_name][1:])
    batch_feed = [tf.reshape(s_d[0], [-1, shape]), tf.reshape(u_d[0], [-1, shape]), s_d[1], u_d[1]]
    batch_feed = [tf.identity(i) for i in batch_feed]
    summary_step = -1 if step % 50 > 0 else step
    with tf.GradientTape() as tape:
        x, bbn, context_ind, feat, pred, adj = model(batch_feed, training=True, step=summary_step)
        img = tf.concat([batch_feed[0], batch_feed[1]], axis=0)
        loss = model.obj(img, s_d[1], x, bbn, context_ind, feat, adj, pred, step=summary_step)
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


def step_val(model: Model, data: Data, step):
    t_d = data.next_test()
    shape = np.prod(DATASET_SHAPE[data.set_name][1:])
    d = tf.reshape(t_d[0], [-1, shape])
    pred = model([d], training=False)

    acc = eval.acc(t_d[1], pred)
    err = 1 - acc
    tf.summary.scalar('val/acc', acc, step=step)
    tf.summary.scalar('val/err', err, step=step)

    return acc


def main():
    max_iter = 150000
    set_name = 'mnist'
    latent_size = 128
    class_num = DATASET_CLASS_COUNT[set_name]
    batch_size = 100
    num_labeled = 100
    l2 = False
    temp = 1
    share_encoder = True
    model = Model(set_name, latent_size, class_num, share_encoder=share_encoder, temp=temp, l2=l2)
    data = Data(set_name, batch_size, num_labeled=num_labeled, label_map_index=0)
    # lr = scheduler.CustomSchedule(latent_size, 2000)
    lr = 1e-4
    opt = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98,
                                   epsilon=1e-9)

    name = 'adjNL{}Share{}Tmp{}l2{}LS{}'.format(num_labeled, share_encoder, temp, l2, latent_size)
    time_string = name + '_' + strftime("%d%b-%H%M", gmtime())
    result_path = os.path.join(ROOT_PATH, 'result', set_name)
    save_path = os.path.join(result_path, 'model', time_string)
    summary_path = os.path.join(result_path, 'log', time_string)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    writer = tf.summary.create_file_writer(summary_path)
    checkpoint = tf.train.Checkpoint(opt=opt, model=model)

    for i in range(max_iter):
        with writer.as_default():
            train_loss = step_train(model, data, opt, i)
            if i == 0:
                print(model.summary())
            if (i + 1) % 200 == 0:
                print('Step: {}, Loss: {}'.format(i, train_loss.numpy()))
                test_acc = step_val(model, data, i)
                print('Hook: {}, acc: {}'.format(i, test_acc.numpy()))

            if (i + 1) % 5000 == 0:
                save_name = os.path.join(save_path, '_' + str(i))
                checkpoint.save(file_prefix=save_name)


if __name__ == '__main__':
    main()
