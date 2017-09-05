# -*- encoding: utf-8 -*-
# Created by han on 17-7-8
import random
import numpy as np
import cPickle


def get_sen_win(sen, win_size, pad_id):
    """
    get window of win_size for each element in the input sentence
    _______________________________________________________________
    Example:
        input  : sen = [w1, w2, w3, w4, w5], win_size = 2, pad_id = 0
        output : [[w1, w2], [w2, w3], [w3, w4], [w4, w5], [w5, 0]]
    """
    win_list = [list(sen)]
    for i in range(1, win_size):
        win_list.append(win_list[0][i:] + [pad_id] * i)
    return zip(*win_list)


class InputData(object):
    """
    data structure feed to the model
    """
    def __init__(self, x, pos1, pos2, slen, y, win=None):
        self.x = x
        self.pos1 = pos1
        self.pos2 = pos2
        self.slen = slen
        self.y = y
        if win is not None:
            self.win = win


class DataLoader(object):
    """
    load data the model needed
    """
    def __init__(self, data_dir, c_feature=False, multi_ins=False, cnn_win_size=0):

        self.c_feature = c_feature
        if c_feature:
            self.embedding = np.load('{}/char_vec.npy'.format(data_dir))

            self.train_y = np.load('{}/s-ins/train_y.npy'.format(data_dir))
            self.train_x = np.load('{}/s-ins/train_char.npy'.format(data_dir))
            self.train_pos1 = np.load('{}/s-ins/train_pos1_c.npy'.format(data_dir))
            self.train_pos2 = np.load('{}/s-ins/train_pos2_c.npy'.format(data_dir))
            self.train_len = np.load('{}/s-ins/train_len_c.npy'.format(data_dir))
            self.train_win = None

            self.test_y = np.load('{}/s-ins/test_y.npy'.format(data_dir))
            self.test_x = np.load('{}/s-ins/test_char.npy'.format(data_dir))
            self.test_pos1 = np.load('{}/s-ins/test_pos1_c.npy'.format(data_dir))
            self.test_pos2 = np.load('{}/s-ins/test_pos2_c.npy'.format(data_dir))
            self.test_len = np.load('{}/s-ins/test_len_c.npy'.format(data_dir))
            self.test_win = None

        else:
            self.embedding = np.load('{}/word_vec.npy'.format(data_dir))

            self.train_y = np.load('{}/s-ins/train_y.npy'.format(data_dir))
            self.train_x = np.load('{}/s-ins/train_word.npy'.format(data_dir))
            self.train_pos1 = np.load('{}/s-ins/train_pos1.npy'.format(data_dir))
            self.train_pos2 = np.load('{}/s-ins/train_pos2.npy'.format(data_dir))
            self.train_len = np.load('{}/s-ins/train_len.npy'.format(data_dir))
            self.train_win = None

            self.test_y = np.load('{}/s-ins/test_y.npy'.format(data_dir))
            self.test_x = np.load('{}/s-ins/test_word.npy'.format(data_dir))
            self.test_pos1 = np.load('{}/s-ins/test_pos1.npy'.format(data_dir))
            self.test_pos2 = np.load('{}/s-ins/test_pos2.npy'.format(data_dir))
            self.test_len = np.load('{}/s-ins/test_len.npy'.format(data_dir))
            self.test_win = None

        self.max_sen_len = len(self.test_x[0])

        if cnn_win_size:
            # window feature used in cnn
            if c_feature:
                with open('./data/char2id.pkl', 'rb') as f:
                    char2id = cPickle.load(f)
                pad_id = char2id['_BLANK']
            else:
                with open('./data/word2id.pkl', 'rb') as f:
                    word2id = cPickle.load(f)
                pad_id = word2id['_BLANK']

            train_x_win, test_x_win = [], []

            # train word
            for sen in self.train_x:
                train_x_win.append(get_sen_win(sen, cnn_win_size, pad_id))
            self.train_win = np.asarray(train_x_win, dtype='int64')

            # test word
            for sen in self.test_x:
                test_x_win.append(get_sen_win(sen, cnn_win_size, pad_id))
            self.test_win = np.asarray(test_x_win, dtype='int64')

        self.multi_ins = multi_ins
        if multi_ins:
            self.train_y_mi = np.load('{}/m-ins/train_y.npy'.format(data_dir))
            self.train_x_mi = np.load('{}/m-ins/train_x.npy'.format(data_dir))
            self.test_y_mi = np.load('{}/m-ins/test_y.npy'.format(data_dir))
            self.test_x_mi = np.load('{}/m-ins/test_x.npy'.format(data_dir))

            train_x, train_len, train_p1, train_p2, train_x_win = [], [], [], [], []
            for idx_list in self.train_x_mi:
                train_x.append(self.train_x[idx_list])
                train_len.append(self.train_len[idx_list])
                train_p1.append(self.train_pos1[idx_list])
                train_p2.append(self.train_pos2[idx_list])
                if cnn_win_size:
                    train_x_win.append(self.train_win[idx_list])

            self.train_x = np.asarray(train_x)
            self.train_len = np.asarray(train_len)
            self.train_pos1 = np.asarray(train_p1)
            self.train_pos2 = np.asarray(train_p2)

            test_x, test_len, test_p1, test_p2, test_x_win = [], [], [], [], []
            for idx_list in self.test_x_mi:
                test_x.append(self.test_x[idx_list])
                test_len.append(self.test_len[idx_list])
                test_p1.append(self.test_pos1[idx_list])
                test_p2.append(self.test_pos2[idx_list])
                if cnn_win_size:
                    test_x_win.append(self.test_win[idx_list])

            self.test_x = np.asarray(test_x)
            self.test_len = np.asarray(test_len)
            self.test_pos1 = np.asarray(test_p1)
            self.test_pos2 = np.asarray(test_p2)

            if cnn_win_size:
                self.train_win = np.asarray(train_x_win)
                self.test_win = np.asarray(test_x_win)

    def get_train_batches(self, batch_size):
        """
        get training data by batch
        """
        if self.multi_ins:
            train_order = range(len(self.train_y_mi))
        else:
            train_order = range(len(self.train_y))

        random.shuffle(train_order)
        batch_num = int(len(train_order) / batch_size)
        for i in range(batch_num):
            batch_order = train_order[i * batch_size: (i + 1) * batch_size]
            if self.train_win:
                batch = InputData(self.train_x[batch_order],
                                  self.train_pos1[batch_order],
                                  self.train_pos2[batch_order],
                                  self.train_len[batch_order],
                                  self.train_y[batch_order],
                                  self.train_win[batch_order])
            else:
                batch = InputData(self.train_x[batch_order],
                                  self.train_pos1[batch_order],
                                  self.train_pos2[batch_order],
                                  self.train_len[batch_order],
                                  self.train_y[batch_order])
            yield batch

    def get_test_batches(self, batch_size):
        """
        get testing data by batch
        """
        if self.multi_ins:
            test_order = range(len(self.test_y_mi))
        else:
            test_order = range(len(self.test_y))

        random.shuffle(test_order)
        batch_num = int(len(test_order) / batch_size)
        for i in range(batch_num):
            batch_order = test_order[i * batch_size: (i + 1) * batch_size]
            if self.test_win:
                batch = InputData(self.test_x[batch_order],
                                  self.test_pos1[batch_order],
                                  self.test_pos2[batch_order],
                                  self.test_len[batch_order],
                                  self.test_y[batch_order],
                                  self.test_win[batch_order])
            else:
                batch = InputData(self.test_x[batch_order],
                                  self.test_pos1[batch_order],
                                  self.test_pos2[batch_order],
                                  self.test_len[batch_order],
                                  self.test_y[batch_order])
            yield batch

    def get_test_all(self):
        """
        get all testing data
        """
        test_data = InputData(
            self.test_x, self.test_pos1, self.test_pos2, self.test_len, self.test_y, self.test_win
        )
        return test_data
