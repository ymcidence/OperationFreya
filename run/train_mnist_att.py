from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
from time import gmtime, strftime
from model.attentional_model import AttentionalModel as Model
from model.attentional_model import step_val, step_train

from util.data.basic_data import BasicData as Data
from util.data.processing import DATASET_CLASS_COUNT
from util import scheduler

from meta import ROOT_PATH


def main(name, max_iter=150000, set_name='mnist', num_labeled=100, batch_size=100, share_encoder=True, restore=None):
    latent_size = 256
    class_num = DATASET_CLASS_COUNT[set_name]

    l2 = False
    temp = 1
    model = Model(set_name, latent_size, class_num, share_encoder=share_encoder, temp=temp, l2=l2)
    data = Data(set_name, batch_size, num_labeled=num_labeled, label_map_index=0)
    lr = scheduler.CustomSchedule(latent_size, 2000)
    # lr = 5e-6
    opt = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98,
                                   epsilon=1e-9)

    # opt = tf.optimizers.RMSprop(lr)

    # name = 'augNL{}Share{}Tmp{}l2{}LS{}'.format(num_labeled, share_encoder, temp, l2, latent_size)

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
    if restore is not None:
        # checkpoint_ = tf.train.Checkpoint(model=model)
        checkpoint.restore(restore)
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
    set_name = 'mnist'
    name = 'not_sharing'
    restore_file = os.path.join(ROOT_PATH, 'result', set_name, 'model', 'consistency_05Oct-1318', '_149999-30')
    share_encoder = False
    main(name=name, max_iter=350000, set_name='mnist', share_encoder=share_encoder, restore=None)
