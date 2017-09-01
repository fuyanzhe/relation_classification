# -*- encoding: utf-8 -*-
# Created by han on 17-8-15


class CnnSetting(object):
    def __init__(self):
        self.win_size = 3
        self.pos_num = 200
        self.pos_size = 5
        self.class_num = 31
        self.sen_len = 100
        self.filter_sizes = [3]
        self.filter_num = 200
        self.learning_rate = None
        self.dropout_rate = 0.5


class RnnSetting(object):
    def __init__(self):
        self.cell = 'gru'
        self.pos_num = 200
        self.pos_size = 5
        self.class_num = 31
        self.sen_len = 100
        self.hidden_size = 200
        self.layers = 1
        self.learning_rate = None
        self.dropout_rate = 0.5
        self.hidden_select = 'avg'


class RnnSetting_SelfAtt(object):
    def __init__(self):
        self.cell = 'gru'
        self.pos_num = 200
        self.pos_size = 5
        self.class_num = 31
        self.sen_len = 100
        self.hidden_size = 200
        self.layers = 1
        self.learning_rate = None
        self.dropout_rate = 0.5
        # self attention hyper parameters
        self.da = 400
        self.r = 31


class RnnMiSetting(object):
    def __init__(self):
        self.cell = 'gru'
        self.pos_num = 200
        self.pos_size = 5
        self.class_num = 31
        self.sen_len = 100
        self.hidden_size = 200
        self.layers = 1
        self.learning_rate = None
        self.dropout_rate = 0.5
        self.bag_num = None
        self.hidden_select = 'last'
