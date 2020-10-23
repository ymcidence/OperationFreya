from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
from time import gmtime, strftime
from model.gumbel_model import GumbelModel as Model
from model.gumbel_model import step_val, step_train

from util.data.basic_data import BasicData as Data
from util.data.processing import DATASET_CLASS_COUNT
# from util import scheduler

from meta import ROOT_PATH


def main(name, batch_size, max_iter=150000, set_name='cifar10', num_labeled=1000, share_encoder=True, restore=None):
    latent_size = 192
    class_num = DATASET_CLASS_COUNT[set_name]

    l2 = False
    temp = 1
    model = Model(set_name, latent_size, class_num, share_encoder=share_encoder, temp=temp, l2=l2)
    data = Data(set_name, batch_size, num_labeled=num_labeled, label_map_index=0)
    # lr = scheduler.CustomSchedule(latent_size, 2000)
    lr = 2e-4
    # opt = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98,
    #                                epsilon=1e-9)
    opt1 = tf.keras.optimizers.Adam(lr)
    opt2 = tf.keras.optimizers.Adam(lr)

    time_string = name + '_' + strftime("%d%b-%H%M", gmtime())
    result_path = os.path.join(ROOT_PATH, 'result', set_name)
    save_path = os.path.join(result_path, 'model', time_string)
    summary_path = os.path.join(result_path, 'log', time_string)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    writer = tf.summary.create_file_writer(summary_path)

    checkpoint = tf.train.Checkpoint(opt1=opt1, opt2=opt2, model=model)
    if restore is not None:
        # checkpoint_ = tf.train.Checkpoint(model=model)
        print('loading checkpoints')
        checkpoint.restore(restore)
    for i in range(max_iter):
        with writer.as_default():
            train_loss = step_train(model, data, opt1, opt2, i)
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
    # noinspection PyUnresolvedReferences
    tf.config.run_functions_eagerly(False)
    set_name = 'cifar10'
    name = 'gumbel4000'
    share_encoder = True
    batch_size = [100, 150]
    num_labeled = 4000
    main(name=name, batch_size=batch_size, max_iter=500000, set_name=set_name, num_labeled=num_labeled,
         share_encoder=share_encoder, restore=None)
