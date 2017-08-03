# -*- encoding: utf-8 -*-
# Created by han on 17-7-8
import numpy as np


class DataLoader(object):
    def __init__(self, data_dir, multi_ins=True):
        self.multi_ins = multi_ins

        print 'reading embeddings'
        self.wordembedding = np.load('./data/vec.npy')

        if self.multi_ins:
            # 多实例模型
            print 'reading training data'
            self.train_y = np.load('./data/small_y.npy')
            self.train_word = np.load('./data/small_word.npy')
            self.train_pos1 = np.load('./data/small_pos1.npy')
            self.train_pos2 = np.load('./data/small_pos2.npy')

            print 'reading testing data'
            self.test_y = np.load('./data/pone_test_y.npy')
            self.test_word = np.load('./data/pone_test_word.npy')
            self.test_pos1 = np.load('./data/pone_test_pos1.npy')
            self.test_pos2 = np.load('./data/pone_test_pos2.npy')
        else:
            # 单实例模型
            print 'reading training data'
            self.train_y = np.load('./data/s-ins/single_train_y.npy')
            self.train_word = np.load('./data/s-ins/single_train_word.npy')
            self.train_pos1 = np.load('./data/s-ins/single_train_pos1.npy')
            self.train_pos2 = np.load('./data/s-ins/single_train_pos2.npy')
            self.train_word = np.load('./data/s-ins/single_train_len.npy')

            print 'reading testing data'
            self.test_y = np.load('./data/s-ins/single_test_y.npy')
            self.test_word = np.load('./data/s-ins/single_test_word.npy')
            self.test_pos1 = np.load('./data/s-ins/single_test_pos1.npy')
            self.test_pos2 = np.load('./data/s-ins/single_test_pos2.npy')
            self.test_word = np.load('./data/s-ins/single_test_len.npy')


    def get_batch(self):
        pass