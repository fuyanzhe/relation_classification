# -*- encoding: utf-8 -*-
# Created by han on 17-8-15


class CNNSetting(object):
    def __init__(self):
        self.win_size = 3
        self.pos_num = 200
        self.pos_size = 5
        self.class_num = 31
        self.sent_len = 100
        self.filter_sizes = [3]
        self.filter_num = 200
        self.learning_rate = None
        self.dropout_rate = 0.5


class RNNSetting(object):
    def __init__(self):
        self.pos_num = 200
        self.pos_size = 5
        self.class_num = 31
        self.sent_len = 100
        self.hidden_size = 100
        self.layers = 1
        self.learning_rate = None
        self.dropout_rate = 0.5


class RNNMiSetting(object):
    def __init__(self):
        self.pos_num = 200
        self.pos_size = 5
        self.class_num = 31
        self.sent_len = 100
        self.hidden_size = 100
        self.layers = 1
        self.learning_rate = None
        self.dropout_rate = 0.5
        self.bag_num = None
