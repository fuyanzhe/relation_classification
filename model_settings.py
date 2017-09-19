# -*- encoding: utf-8 -*-
# Created by han on 17-8-15


class CnnSetting(object):
    def __init__(self):
        self.pos_num = 200
        self.pos_size = 5
        self.class_num = 31
        self.sen_len = 100
        self.ent_len = None
        # cnn
        self.filter_sizes = [2, 4, 6, 10, 20]
        self.filter_num = 200
        # learning settings
        self.optimizer = 'adam'
        self.learning_rate = 0.001
        self.dropout_rate = 0.5


class DeepCnnSetting(object):
    def __init__(self):
        # data settings
        self.pos_num = 200
        self.pos_size = 5
        self.class_num = 31
        self.sen_len = 100
        self.ent_len = None
        # conv layers
        self.filter_sizes = [3, 5, 7, 7]
        self.filter_num = [256] * len(self.filter_sizes)
        # max pool layers, 0 when max pool is not used
        self.max_pool_sizes = [(2, 1), (3, 1), (5, 1), (5, 1)]
        assert len(self.filter_sizes) == len(self.max_pool_sizes)
        # full connection layers
        self.fc_sizes = [self.class_num]
        # drop out
        self.dropout_mask = [1, 0, 0, 0, 1]
        assert len(self.dropout_mask) == len(self.filter_sizes) + len(self.fc_sizes)
        # optimizer
        self.optimizer = 'adam'
        # learning settings
        self.learning_rate = 0.001
        self.dropout_rate = 0.5


class RnnSetting(object):
    def __init__(self):
        self.pos_num = 200
        self.pos_size = 5
        self.class_num = 31
        self.sen_len = 100
        self.ent_len = None
        # rnn
        self.cell = 'gru'
        self.hidden_size = 200
        self.layers = 1
        self.hidden_select = 'avg'
        # optimizer
        self.optimizer = 'adam'
        self.learning_rate = 0.001
        self.dropout_rate = 0.5


class RnnSelfAttSetting(object):
    def __init__(self):
        self.pos_num = 200
        self.pos_size = 5
        self.class_num = 31
        self.sen_len = 100
        self.ent_len = None
        # rnn
        self.cell = 'gru'
        self.hidden_size = 200
        self.layers = 1
        # self attention hyper parameters
        self.da = 400
        self.r = 31
        # optimizer
        self.optimizer = 'adam'
        self.learning_rate = 0.001
        self.dropout_rate = 0.5


class RnnMiSetting(object):
    def __init__(self):
        self.pos_num = 200
        self.pos_size = 5
        self.class_num = 31
        self.sen_len = 100
        self.ent_len = None
        # rnn
        self.cell = 'gru'
        self.hidden_size = 200
        self.layers = 1
        self.hidden_select = 'avg'
        # optimizer
        self.optimizer = 'adam'
        self.learning_rate = 0.001
        self.dropout_rate = 0.5
        # multi-instance
        self.bag_num = None


class RnnResSetting(object):
    def __init__(self):
        self.pos_num = 200
        self.pos_size = 5
        self.class_num = 31
        self.sen_len = 100
        self.ent_len = None
        # rnn
        self.cell = 'gru'
        self.hidden_size = 200
        self.layers = 2
        # optimizer
        self.optimizer = 'adam'
        self.learning_rate = 0.001
        self.dropout_rate = 0.5


class RnnEntSetting(object):
    def __init__(self):
        self.pos_num = 200
        self.pos_size = 5
        self.class_num = 31
        self.sen_len = 100
        self.ent_len = None
        # rnn
        self.cell = 'gru'
        self.hidden_size_sen = 200
        self.hidden_size_ent = 200
        self.layers = 1
        self.hidden_select = 'avg'
        # optimizer
        self.optimizer = 'adam'
        self.learning_rate = 0.001
        self.dropout_rate = 0.5


class RnnAttEntSetting(object):
    def __init__(self):
        self.pos_num = 200
        self.pos_size = 5
        self.class_num = 31
        self.sen_len = 100
        self.ent_len = None
        # rnn
        self.cell = 'gru'
        self.hidden_size_sen = 200
        self.hidden_size_ent = 200
        self.layers = 1
        # optimizer
        self.optimizer = 'adam'
        self.learning_rate = 0.001
        self.dropout_rate = 0.5


class RnnCnnEntSetting(object):
    def __init__(self):
        self.pos_num = 200
        self.pos_size = 5
        self.class_num = 31
        self.sen_len = 100
        self.ent_len = None
        # rnn
        self.cell = 'gru'
        self.hidden_size_sen = 200
        self.hidden_size_ent = 200
        self.layers = 1
        self.hidden_select = 'avg'
        # cnn
        self.filter_sizes = [1, 2, 3, 4, 5]
        self.filter_num = 200
        # optimizer
        self.optimizer = 'adam'
        self.learning_rate = 0.001
        self.dropout_rate = 0.5