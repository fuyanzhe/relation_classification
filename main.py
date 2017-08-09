# -*- encoding: utf-8 -*-
# Created by han on 17-7-8
from models import *
from data_loader import DataLoader


class cnn_setting(object):
    pass


class rnn_setting(object):
    pass


def main():
    data_loader = DataLoader('./data', multi_ins=False)
    batches = data_loader.get_train_batches(batch_size=1000)
    for batch in batches:
        print len(batch), len(batch['word']), len(batch['pos1']), len(batch['pos2']), len(batch['len']), len(batch['y'])
        print batch['word'][0]
        print batch['pos1'][0]
        print batch['pos2'][0]
        print batch['len'][0]
        print batch['y'][0]
    pass

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            initializer = tf.contrib.layers.xavier_initializer()


if __name__ == '__main__':
    main()