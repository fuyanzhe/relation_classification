# -*- encoding: utf-8 -*-
# Created by han on 17-7-8

import random
import numpy as np
import cPickle


class InputData(object):
    """
    data structure feed to the model
    """
    def __init__(self, x, pos1, pos2, slen, e1, e2, e1_len, e2_len, y):
        self.x = x
        self.pos1 = pos1
        self.pos2 = pos2
        self.slen = slen
        self.e1 = e1
        self.e2 = e2
        self.e1_len = e1_len
        self.e2_len = e2_len
        self.y = y


class DataLoader(object):
    """
    load data the model needed
    """
    def __init__(self, data_dir, c_feature=False, multi_ins=False):

        self.c_feature = c_feature
        if c_feature:
            self.embedding = np.load('{}/char_vec.npy'.format(data_dir))

            self.train_y = np.load('{}/s-ins/train_y.npy'.format(data_dir))
            self.train_x = np.load('{}/s-ins/train_char.npy'.format(data_dir))
            self.train_pos1 = np.load('{}/s-ins/train_pos1_c.npy'.format(data_dir))
            self.train_pos2 = np.load('{}/s-ins/train_pos2_c.npy'.format(data_dir))
            self.train_len = np.load('{}/s-ins/train_len_c.npy'.format(data_dir))
            self.train_e1 = np.load('{}/s-ins/train_e1_c.npy'.format(data_dir))
            self.train_e2 = np.load('{}/s-ins/train_e2_c.npy'.format(data_dir))
            self.train_e1_len = np.load('{}/s-ins/train_e1_len_c.npy'.format(data_dir))
            self.train_e2_len = np.load('{}/s-ins/train_e2_len_c.npy'.format(data_dir))

            self.test_y = np.load('{}/s-ins/test_y.npy'.format(data_dir))
            self.test_x = np.load('{}/s-ins/test_char.npy'.format(data_dir))
            self.test_pos1 = np.load('{}/s-ins/test_pos1_c.npy'.format(data_dir))
            self.test_pos2 = np.load('{}/s-ins/test_pos2_c.npy'.format(data_dir))
            self.test_len = np.load('{}/s-ins/test_len_c.npy'.format(data_dir))
            self.test_e1 = np.load('{}/s-ins/test_e1_c.npy'.format(data_dir))
            self.test_e2 = np.load('{}/s-ins/test_e2_c.npy'.format(data_dir))
            self.test_e1_len = np.load('{}/s-ins/test_e1_len_c.npy'.format(data_dir))
            self.test_e2_len = np.load('{}/s-ins/test_e2_len_c.npy'.format(data_dir))

        else:
            self.embedding = np.load('{}/word_vec.npy'.format(data_dir))

            self.train_y = np.load('{}/s-ins/train_y.npy'.format(data_dir))
            self.train_x = np.load('{}/s-ins/train_word.npy'.format(data_dir))
            self.train_pos1 = np.load('{}/s-ins/train_pos1.npy'.format(data_dir))
            self.train_pos2 = np.load('{}/s-ins/train_pos2.npy'.format(data_dir))
            self.train_len = np.load('{}/s-ins/train_len.npy'.format(data_dir))
            self.train_e1 = np.load('{}/s-ins/train_e1.npy'.format(data_dir))
            self.train_e2 = np.load('{}/s-ins/train_e2.npy'.format(data_dir))
            self.train_e1_len = np.load('{}/s-ins/train_e1_len.npy'.format(data_dir))
            self.train_e2_len = np.load('{}/s-ins/train_e2_len.npy'.format(data_dir))

            self.test_y = np.load('{}/s-ins/test_y.npy'.format(data_dir))
            self.test_x = np.load('{}/s-ins/test_word.npy'.format(data_dir))
            self.test_pos1 = np.load('{}/s-ins/test_pos1.npy'.format(data_dir))
            self.test_pos2 = np.load('{}/s-ins/test_pos2.npy'.format(data_dir))
            self.test_len = np.load('{}/s-ins/test_len.npy'.format(data_dir))
            self.test_e1 = np.load('{}/s-ins/test_e1.npy'.format(data_dir))
            self.test_e2 = np.load('{}/s-ins/test_e2.npy'.format(data_dir))
            self.test_e1_len = np.load('{}/s-ins/test_e1_len.npy'.format(data_dir))
            self.test_e2_len = np.load('{}/s-ins/test_e2_len.npy'.format(data_dir))

        self.max_sen_len = len(self.test_x[0])
        self.max_ent_len = len(self.test_e1[0])

        self.multi_ins = multi_ins
        if multi_ins:
            self.train_y_mi = np.load('{}/m-ins/train_y.npy'.format(data_dir))
            self.train_x_mi = np.load('{}/m-ins/train_x.npy'.format(data_dir))
            self.test_y_mi = np.load('{}/m-ins/test_y.npy'.format(data_dir))
            self.test_x_mi = np.load('{}/m-ins/test_x.npy'.format(data_dir))

            train_x, train_len, train_p1, train_p2 = [], [], [], []
            train_e1, train_e2, train_e1_len, train_e2_len = [], [], [], []
            for idx_list in self.train_x_mi:
                train_x.append(self.train_x[idx_list])
                train_len.append(self.train_len[idx_list])
                train_p1.append(self.train_pos1[idx_list])
                train_p2.append(self.train_pos2[idx_list])
                train_e1.append(self.train_e1[idx_list])
                train_e2.append(self.train_e2[idx_list])
                train_e1_len.append(self.train_e1_len[idx_list])
                train_e2_len.append(self.train_e2_len[idx_list])

            self.train_x = np.asarray(train_x)
            self.train_len = np.asarray(train_len)
            self.train_pos1 = np.asarray(train_p1)
            self.train_pos2 = np.asarray(train_p2)
            self.train_e1 = np.asarray(train_e1)
            self.train_e2 = np.asarray(train_e2)
            self.train_e1_len = np.asarray(train_e1_len)
            self.train_e2_len = np.asarray(train_e2_len)

            test_x, test_len, test_p1, test_p2 = [], [], [], []
            test_e1, test_e2, test_e1_len, test_e2_len = [], [], [], []
            for idx_list in self.test_x_mi:
                test_x.append(self.test_x[idx_list])
                test_len.append(self.test_len[idx_list])
                test_p1.append(self.test_pos1[idx_list])
                test_p2.append(self.test_pos2[idx_list])
                test_e1.append(self.test_e1[idx_list])
                test_e2.append(self.test_e2[idx_list])
                test_e1_len.append(self.test_e1_len[idx_list])
                test_e2_len.append(self.test_e2_len[idx_list])

            self.test_x = np.asarray(test_x)
            self.test_len = np.asarray(test_len)
            self.test_pos1 = np.asarray(test_p1)
            self.test_pos2 = np.asarray(test_p2)
            self.test_e1 = np.asarray(test_e1)
            self.test_e2 = np.asarray(test_e2)
            self.test_e1_len = np.asarray(test_e1_len)
            self.test_e2_len = np.asarray(test_e2_len)

    def compute_pcnn_pool_mask(self, x, s_len, pos1,  pos2):
        """
        used in pcnn, get piece wise max pooling mask
        """
        s_num, s_max_l = x.shape
        pos1_0 = pos1[:0]
        pos2_0 = pos1[:0]
        p_e1 = s_max_l - pos1_0 + 1
        p_e2 = s_max_l - pos2_0 + 1
        mask = np.zeros((s_num, 3, s_max_l), dtype=np.float32)
        for i in range(s_num):
            p1 = p_e1[i]
            p2 = p_e2[i]
            if p1 <= p2:
                mask[i, 0, :p1] = 1
                mask[i, 1, p1:p2] = 1
                mask[i, 2, p2:s_max_l] = 1
            else:
                mask[i, 0, :p2] = 1
                mask[i, 1, p2:p1] = 1
                mask[i, 2, p1:s_max_l] = 1
        return mask

    def get_train_batches(self, batch_size):
        """
        get training data by batch
        """
        if self.multi_ins:
            train_order = range(len(self.train_y_mi))
            select_y = self.train_y_mi
        else:
            train_order = range(len(self.train_y))
            select_y = self.train_y

        random.shuffle(train_order)
        batch_num = int(len(train_order) / batch_size)
        for i in range(batch_num):
            batch_order = train_order[i * batch_size: (i + 1) * batch_size]
            batch = InputData(self.train_x[batch_order],
                              self.train_pos1[batch_order],
                              self.train_pos2[batch_order],
                              self.train_len[batch_order],
                              self.train_e1[batch_order],
                              self.train_e2[batch_order],
                              self.train_e1_len[batch_order],
                              self.train_e2_len[batch_order],
                              select_y[batch_order])
            yield batch

    def get_test_batches(self, batch_size):
        """
        get testing data by batch
        """
        if self.multi_ins:
            test_order = range(len(self.test_y_mi))
            select_y = self.test_y_mi
        else:
            test_order = range(len(self.test_y))
            select_y = self.test_y

        random.shuffle(test_order)
        batch_num = int(len(test_order) / batch_size)
        for i in range(batch_num):
            batch_order = test_order[i * batch_size: (i + 1) * batch_size]
            batch = InputData(self.test_x[batch_order],
                              self.test_pos1[batch_order],
                              self.test_pos2[batch_order],
                              self.test_len[batch_order],
                              self.test_e1[batch_order],
                              self.test_e2[batch_order],
                              self.test_e1_len[batch_order],
                              self.test_e2_len[batch_order],
                              select_y[batch_order])
            yield batch

    def get_test_all(self):
        """
        get all testing data
        """
        select_y = self.test_y_mi if self.multi_ins else self.test_y
        test_data = InputData(
            self.test_x, self.test_pos1, self.test_pos2, self.test_len,
            self.test_e1, self.test_e2, self.test_e1_len, self.test_e2_len,
            select_y
        )
        return test_data
