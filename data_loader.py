# -*- encoding: utf-8 -*-
# Created by han on 17-7-8
import random
import numpy as np
import cPickle


class InputData(object):
    def __init__(self, word, pos1, pos2, slen, y):
        self.word = word
        self.pos1 = pos1
        self.pos2 = pos2
        self.slen = slen
        self.y = y


class DataLoader(object):
    def __init__(self, data_dir, multi_ins=False, cnn_win_size=0):
        self.multi_ins = multi_ins

        print 'reading embeddings'
        self.wordembedding = np.load('{}/vec.npy'.format(data_dir))

        if self.multi_ins:
            # 多实例模型
            print 'reading training data'
            self.train_y = np.load('{}/m-ins/train_y.npy'.format(data_dir))
            self.train_word = np.load('{}/m-ins/train_word.npy'.format(data_dir))
            self.train_pos1 = np.load('{}/m-ins/train_pos1.npy'.format(data_dir))
            self.train_pos2 = np.load('{}/m-ins/train_pos2.npy'.format(data_dir))
            self.train_len = np.load('{}/m-ins/train_len.npy'.format(data_dir))

            print 'reading testing data'
            self.test_y = np.load('{}/m-ins/testall_y.npy'.format(data_dir))
            self.test_word = np.load('{}/m-ins/testall_word.npy'.format(data_dir))
            self.test_pos1 = np.load('{}/m-ins/testall_pos1.npy'.format(data_dir))
            self.test_pos2 = np.load('{}/m-ins/testall_pos2.npy'.format(data_dir))
            self.test_len = np.load('{}/m-ins/testall_len.npy'.format(data_dir))

            print 'reading single testing data'
            self.test_y_single = np.load('{}/s-ins/test_y.npy'.format(data_dir))
            self.test_word_single = np.load('{}/s-ins/test_word.npy'.format(data_dir))
            self.test_pos1_single = np.load('{}/s-ins/test_pos1.npy'.format(data_dir))
            self.test_pos2_single = np.load('{}/s-ins/test_pos2.npy'.format(data_dir))
            self.test_len_single = np.load('{}/s-ins/test_len.npy'.format(data_dir))

        else:
            # 单实例模型
            print 'reading training data'
            self.train_y = np.load('{}/s-ins/train_y.npy'.format(data_dir))
            self.train_word = np.load('{}/s-ins/train_word.npy'.format(data_dir))
            self.train_pos1 = np.load('{}/s-ins/train_pos1.npy'.format(data_dir))
            self.train_pos2 = np.load('{}/s-ins/train_pos2.npy'.format(data_dir))
            self.train_len = np.load('{}/s-ins/train_len.npy'.format(data_dir))

            print 'reading testing data'
            self.test_y = np.load('{}/s-ins/test_y.npy'.format(data_dir))
            self.test_word = np.load('{}/s-ins/test_word.npy'.format(data_dir))
            self.test_pos1 = np.load('{}/s-ins/test_pos1.npy'.format(data_dir))
            self.test_pos2 = np.load('{}/s-ins/test_pos2.npy'.format(data_dir))
            self.test_len = np.load('{}/s-ins/test_len.npy'.format(data_dir))

        if cnn_win_size:
            print 'generating window features...'
            with open('./data/word2id.pkl', 'rb') as f:
                word2id = cPickle.load(f)
            pad_id = word2id['_BLANK']

            def get_sen_win(sen, win_size, pad_id):
                win_list = [list(sen)]
                for i in range(1, win_size):
                    win_list.append(win_list[0][i:] + [pad_id] * i)
                return zip(*win_list)

            if multi_ins:
                train_word_win, test_word_win, test_word_win_single = [], [], []

                # train multi-ins
                for bag in self.train_word:
                    bag_win = []
                    for sen in bag:
                        bag_win.append(get_sen_win(sen, cnn_win_size, pad_id))
                    train_word_win.append(bag_win)
                self.train_word = np.asarray(train_word_win)

                # test multi-ins
                for bag in self.test_word:
                    bag_win = []
                    for sen in bag:
                        bag_win.append(get_sen_win(sen, cnn_win_size, pad_id))
                    test_word_win.append(bag_win)
                self.test_word = np.asarray(test_word_win)

                # test single-ins
                for sen in self.test_word_single:
                    test_word_win_single.append(get_sen_win(sen, cnn_win_size, pad_id))
                self.test_word_single = np.asarray(test_word_win_single, dtype='int64')

            else:
                train_word_win, test_word_win = [], []

                # train word
                for sen in self.train_word:
                    train_word_win.append(get_sen_win(sen, cnn_win_size, pad_id))
                self.train_word = np.asarray(train_word_win, dtype='int64')

                # test word
                for sen in self.test_word:
                    test_word_win.append(get_sen_win(sen, cnn_win_size, pad_id))
                self.test_word = np.asarray(test_word_win, dtype='int64')

        assert len(self.train_y) == len(self.train_len) == len(self.train_word) == len(self.train_pos1) == len(
            self.train_pos2)
        assert len(self.test_y) == len(self.test_len) == len(self.test_word) == len(self.test_pos1) == len(
            self.test_pos2)

    def get_train_batches(self, batch_size):
        train_order = range(len(self.train_y))
        random.shuffle(train_order)
        batch_num = int(len(train_order) / batch_size)
        for i in range(batch_num):
            batch_order = train_order[i * batch_size: (i + 1) * batch_size]
            batch = InputData(self.train_word[batch_order],
                              self.train_pos1[batch_order],
                              self.train_pos2[batch_order],
                              self.train_len[batch_order],
                              self.train_y[batch_order])
            yield batch

    def get_test_data(self, use_neg=True):
        if self.multi_ins:
            test_data = InputData(self.test_word, self.test_pos1, self.test_pos2, self.test_len, self.test_y)
            test_data_single = InputData(
                self.test_word_single, self.test_pos1_single, self.test_pos2_single,
                self.test_len_single, self.test_y_single
            )
            return test_data, test_data_single
        else:
            test_data = InputData(self.test_word, self.test_pos1, self.test_pos2, self.test_len, self.test_y)
            return test_data

    def get_test_batches(self, batch_size, use_single=True):
        if self.multi_ins:
            if use_single:
                test_order = range(len(self.test_y_single))
                random.shuffle(test_order)
                batch_num = int(len(test_order) / batch_size)
                for i in range(batch_num):
                    batch_order = test_order[i * batch_size: (i + 1) * batch_size]
                    batch = InputData(self.test_word_single[batch_order],
                                      self.test_pos1_single[batch_order],
                                      self.test_pos2_single[batch_order],
                                      self.test_len_single[batch_order],
                                      self.test_y_single[batch_order])
                    yield batch
            else:
                test_order = range(len(self.test_y))
                random.shuffle(test_order)
                batch_num = int(len(test_order) / batch_size)
                for i in range(batch_num):
                    batch_order = test_order[i * batch_size: (i + 1) * batch_size]
                    batch = InputData(self.test_word[batch_order],
                                      self.test_pos1[batch_order],
                                      self.test_pos2[batch_order],
                                      self.test_len[batch_order],
                                      self.test_y[batch_order])
                    yield batch
        else:
            test_order = range(len(self.test_y))
            random.shuffle(test_order)
            batch_num = int(len(test_order) / batch_size)
            for i in range(batch_num):
                batch_order = test_order[i * batch_size: (i + 1) * batch_size]
                batch = InputData(self.test_word[batch_order],
                                  self.test_pos1[batch_order],
                                  self.test_pos2[batch_order],
                                  self.test_len[batch_order],
                                  self.test_y[batch_order])
                yield batch
