# -*- encoding: utf-8 -*-
# Created by han on 17-7-8
import random
import numpy as np


class DataLoader(object):
    def __init__(self, data_dir, multi_ins=True):
        self.multi_ins = multi_ins

        print 'reading embeddings'
        self.wordembedding = np.load('{}/vec.npy'.format(data_dir))

        if self.multi_ins:
            # 多实例模型
            print 'reading training data'
            self.train_y = np.load('{}/m-ins/small_y.npy'.format(data_dir))
            self.train_word = np.load('{}/m-ins/small_word.npy'.format(data_dir))
            self.train_pos1 = np.load('{}/m-ins/small_pos1.npy'.format(data_dir))
            self.train_pos2 = np.load('{}/m-ins/small_pos2.npy'.format(data_dir))
            self.train_len = np.load('{}/m-ins/train_len.npy'.format(data_dir))

            print 'reading testing data'
            self.test_y = np.load('{}/m-ins/pall_test_y.npy'.format(data_dir))
            self.test_word = np.load('{}/m-ins/pall_test_word.npy'.format(data_dir))
            self.test_pos1 = np.load('{}/m-ins/pall_test_pos1.npy'.format(data_dir))
            self.test_pos2 = np.load('{}/m-ins/pall_test_pos2.npy'.format(data_dir))
            self.test_len = np.load('{}/m-ins/pall_test_len.npy'.format(data_dir))

        else:
            # 单实例模型
            print 'reading training data'
            self.train_y = np.load('{}/s-ins/single_train_y.npy'.format(data_dir))
            self.train_word = np.load('{}/s-ins/single_train_word.npy'.format(data_dir))
            self.train_pos1 = np.load('{}/s-ins/single_train_pos1.npy'.format(data_dir))
            self.train_pos2 = np.load('{}/s-ins/single_train_pos2.npy'.format(data_dir))
            self.train_len = np.load('{}/s-ins/single_train_len.npy'.format(data_dir))

            print 'reading testing data'
            self.test_y = np.load('{}/s-ins/single_test_y.npy'.format(data_dir))
            self.test_word = np.load('{}/s-ins/single_test_word.npy'.format(data_dir))
            self.test_pos1 = np.load('{}/s-ins/single_test_pos1.npy'.format(data_dir))
            self.test_pos2 = np.load('{}/s-ins/single_test_pos2.npy'.format(data_dir))
            self.test_len = np.load('{}/s-ins/single_test_len.npy'.format(data_dir))

        assert len(self.train_y) == len(self.train_len) == len(self.train_word) == len(self.train_pos1) == len(
            self.train_pos2)
        assert len(self.test_y) == len(self.test_len) == len(self.test_word) == len(self.test_pos1) == len(
            self.test_pos2)

    def get_train_batches(self, batch_size):
        if self.multi_ins:
            pass
        else:
            train_order = range(len(self.train_y))
            random.shuffle(train_order)
            batch_num = int(len(train_order) / batch_size)
            for i in range(batch_num):
                batch= dict()
                batch_order = train_order[i * batch_size: (i + 1) * batch_size]
                batch['word'] = self.train_word[batch_order]
                batch['pos1'] = self.train_pos1[batch_order]
                batch['pos2'] = self.train_pos2[batch_order]
                batch['len'] = self.train_len[batch_order]
                batch['y'] = self.train_y[batch_order]
                yield batch

