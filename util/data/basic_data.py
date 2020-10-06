from __future__ import division, print_function, absolute_import, unicode_literals

import tensorflow as tf
from util.data import processing


class BasicData(object):
    def __init__(self, set_name, batch_size=256, num_labeled=100, label_map_index=0, aug=True):
        self.set_name = set_name
        self.batch_size = batch_size
        self.num_labeled = num_labeled
        self.label_map_index = label_map_index
        self.aug = aug
        if isinstance(batch_size, list):
            self.s_size = batch_size[0]
            self.u_size = batch_size[1]
            self.batch_size = self.s_size + self.u_size
        else:
            s_size = batch_size // 3
            self.s_size = s_size if s_size <= 100 else 100
            self.u_size = batch_size - self.s_size
        self.t_size = self.u_size + self.s_size
        self.d_s_train, self.d_u_train = self._training_data()
        self.d_test = self._test_data()

    def _f_semi_supervised_filtering(self):
        """
        :return: a filtering function to separate labeled-unlabeled data for Dataset objects
        """
        label_map_name = "label_map_count_{}_index_{}".format(self.num_labeled, self.label_map_index)
        label_table_labeled = processing.construct_label_table(self.set_name, label_map_name)
        label_table_unlabeled = processing.construct_label_table(self.set_name, label_map_name, labeled=False)

        def f_labeled(_image, _label, _fkey):
            return label_table_labeled.lookup(_fkey)

        def f_unlabeled(_image, _label, _fkey):
            return label_table_unlabeled.lookup(_fkey)

        return f_labeled, f_unlabeled

    def _training_data(self):
        parser = processing.construct_parser(self.set_name)

        filter_1, filter_2 = self._f_semi_supervised_filtering()

        s_data = tf.data.TFRecordDataset(processing.get_filenames(self.set_name, 'train')
                                         ).map(parser, num_parallel_calls=8).prefetch(50)
        u_data = tf.data.TFRecordDataset(processing.get_filenames(self.set_name, 'train')
                                         ).map(parser, num_parallel_calls=8).prefetch(50)

        s_data = s_data.filter(filter_1)
        u_data = u_data.filter(filter_2)

        s_data = s_data.cache().repeat().shuffle(self.num_labeled).batch(self.s_size)
        u_data = u_data.cache().repeat().shuffle(20000).batch(self.u_size)

        return iter(s_data), iter(u_data)

    def _test_data(self):
        parser = processing.construct_parser(self.set_name)
        test_data = tf.data.TFRecordDataset(processing.get_filenames(self.set_name, 'test')
                                            ).map(parser, num_parallel_calls=8).prefetch(50)
        return iter(test_data.cache().repeat().shuffle(10000).batch(self.t_size))

    def next_train(self):
        return next(self.d_s_train), next(self.d_u_train)

    def next_test(self):
        return next(self.d_test)


if __name__ == '__main__':
    import numpy as np

    data = BasicData('mnist', 128)
    a = data.next_train()
    for i in range(10):
        print(i, np.sum(np.where(a[0][1].numpy() == i)))
