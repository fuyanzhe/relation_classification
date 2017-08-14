# -*- encoding: utf-8 -*-
# Created by han on 17-7-11
import tensorflow as tf


class CNNSetting(object):
    def __init__(self):
        self.pos_num = 200
        self.pos_size = 5
        self.class_num = 31
        self.sent_len = 100
        self.filter_sizes = [3]
        self.filter_num = 200
        self.learning_rate = None


class LSTMSetting(object):
    def __init__(self):
        self.pos_num = 200
        self.pos_size = 5
        self.class_num = 31
        self.sent_len = 100
        self.hidden_size = 100
        self.layers = 1
        self.learning_rate = None


class BiGRUSetting(object):
    def __init__(self):
        self.pos_num = 200
        self.pos_size = 5
        self.class_num = 31
        self.sent_len = 100
        self.hidden_size = 100
        self.layers = 1
        self.learning_rate = None


class CNN(object):
    def __init__(self, word_embedding, setting):
        # model name
        self.model_name = 'CNN'

        # embedding matrix
        self.embed_matrix_word = tf.get_variable(
            'embed_matrix_word', word_embedding.shape,
            initializer=tf.constant_initializer(word_embedding)
        )
        self.embed_size_word = int(self.embed_matrix_word.get_shape()[1])
        self.embed_matrix_pos1 = tf.get_variable('embed_matrix_pos1', [setting.pos_num, setting.pos_size])
        self.embed_matrix_pos2 = tf.get_variable('embed_matrix_pos2', [setting.pos_num, setting.pos_size])
        self.embed_size_pos = setting.pos_size

        # max sentence length
        self.max_sentence_len = setting.sent_len

        # filter number
        self.filter_sizes = setting.filter_sizes
        self.filter_num = setting.filter_num

        # number of classes
        self.class_num = setting.class_num

        # inputs
        self.input_words = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_words')
        self.input_labels = tf.placeholder(tf.int32, [None, self.class_num], name='labels')

        # position feature
        self.input_pos1 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos1')
        self.input_pos2 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos2')

        # dropout keep probability
        self.dropout_keep_rate = tf.placeholder(tf.float32, name="dropout_keep_rate")

        # learning rate
        self.learning_rate = setting.learning_rate

        # embedded
        self.emb_word = tf.nn.embedding_lookup(self.embed_matrix_word, self.input_words)
        self.emb_pos1 = tf.nn.embedding_lookup(self.embed_matrix_pos1, self.input_pos1)
        self.emb_pos2 = tf.nn.embedding_lookup(self.embed_matrix_pos2, self.input_pos2)

        # concat embeddings
        self.emb_all = tf.concat([self.emb_word, self.emb_pos1, self.emb_pos2], 2)
        self.emb_all_expanded = tf.expand_dims(self.emb_all, -1)

        # states and outputs
        with tf.name_scope('sentence_encoder'):
            self.outputs = self.sentence_encoder()

        # softmax
        with tf.name_scope('softmax'):
            self.softmax_w = tf.get_variable('softmax_W', [self.filter_num * len(self.filter_sizes), self.class_num])
            self.softmax_b = tf.get_variable('softmax_b', [self.class_num])
            self.softmax_pred = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b
            self.softmax_res = tf.nn.softmax(self.softmax_pred)

        # get max softmax predict result of each relation
        self.maxres_by_rel = tf.reduce_max(self.softmax_res, 0)

        # class label
        self.class_label = tf.argmax(self.softmax_res, 1)

        # choose the min loss instance index
        self.instance_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.softmax_pred, labels=self.input_labels)
        self.min_loss_idx = tf.argmin(self.instance_loss, 0)

        # model loss
        self.model_loss = tf.reduce_mean(self.instance_loss)

        # optimizer
        if self.learning_rate:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.model_loss)
        else:
            self.optimizer = tf.train.AdamOptimizer().minimize(self.model_loss)

        # saver
        self.saver = tf.train.Saver(tf.all_variables())

    def sentence_encoder(self):
        # convolution and max pooling
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # convolution layer
                filter_shape = [
                    filter_size, self.embed_size_word + 2 * self.embed_size_pos, 1, self.filter_num
                ]

                w = tf.get_variable('W', filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
                b = tf.get_variable('b', [self.filter_num], initializer=tf.constant_initializer(0.1))
                conv = tf.nn.conv2d(self.emb_all_expanded, w, strides=[1, 1, 1, 1], padding='VALID', name='conv')

                # Apply none linearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # Max pooling over the outputs
                pooled = tf.nn.max_pool(
                    h, ksize=[1, self.max_sentence_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID', name='conv'
                )
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.filter_num * len(self.filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Add dropout
        h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_rate)

        return h_drop

    def fit(self, session, input_data, dropout_keep_rate):
        feed_dict = {self.input_words: input_data.word,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: dropout_keep_rate
                     }
        session.run(self.optimizer, feed_dict=feed_dict)
        model_loss = session.run(self.model_loss, feed_dict=feed_dict)
        return model_loss

    def predict(self, session, input_data):
        feed_dict = {self.input_words: input_data.word,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: 1}
        model_loss, label_pred = session.run([self.model_loss, self.class_label], feed_dict=feed_dict)
        return model_loss, label_pred


class LSTM(object):
    def __init__(self, word_embedding, setting):
        # model name
        self.model_name = 'LSTM'

        # settings
        self.max_sentence_len = setting.sent_len
        self.hidden_size = setting.hidden_size
        self.class_num = setting.class_num
        self.pos_num = setting.pos_num
        self.pos_size = setting.pos_size
        self.learning_rate = setting.learning_rate

        # embedding matrix
        self.embed_matrix_word = tf.get_variable(
            'embed_matrix_word', word_embedding.shape,
            initializer=tf.constant_initializer(word_embedding)
        )
        self.embed_size_word = int(self.embed_matrix_word.get_shape()[1])
        self.embed_matrix_pos1 = tf.get_variable('embed_matrix_pos1', [self.pos_num, self.pos_size])
        self.embed_matrix_pos2 = tf.get_variable('embed_matrix_pos2', [self.pos_num, self.pos_size])

        # inputs
        self.input_words = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_words')
        self.input_labels = tf.placeholder(tf.int32, [None, self.class_num], name='labels')

        # position feature
        self.input_pos1 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos1')
        self.input_pos2 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos2')

        # dropout keep probability
        self.dropout_keep_rate = tf.placeholder(tf.float32, name="dropout_keep_rate")

        # embedded
        self.emb_word = tf.nn.embedding_lookup(self.embed_matrix_word, self.input_words)
        self.emb_pos1 = tf.nn.embedding_lookup(self.embed_matrix_pos1, self.input_pos1)
        self.emb_pos2 = tf.nn.embedding_lookup(self.embed_matrix_pos2, self.input_pos2)

        # concat embeddings
        self.emb_all = tf.concat([self.emb_word, self.emb_pos1, self.emb_pos2], 2)
        self.emb_all_us = tf.unstack(self.emb_all, num=self.max_sentence_len, axis = 1)

        # states and outputs
        with tf.name_scope('sentence_encoder'):
            # cell
            # self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=0.0, state_is_tuple=True)
            self.lstm_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            self.lstm_cell = tf.nn.rnn_cell.DropoutWrapper(self.lstm_cell, output_keep_prob=self.dropout_keep_rate)

            # rnn
            self.outputs, self.states = tf.contrib.rnn.static_rnn(self.lstm_cell, self.emb_all_us, dtype=tf.float32)

        self.output_final = self.outputs[-1]

        # softmax
        with tf.name_scope('softmax'):
            self.softmax_w = tf.get_variable('softmax_W', [self.hidden_size, self.class_num])
            self.softmax_b = tf.get_variable('softmax_b', [self.class_num])
            self.softmax_pred = tf.matmul(self.output_final, self.softmax_w) + self.softmax_b
            self.softmax_res = tf.nn.softmax(self.softmax_pred)

        # get max softmax predict result of each relation
        self.maxres_by_rel = tf.reduce_max(self.softmax_res, 0)

        # class label
        self.class_label = tf.argmax(self.softmax_res, 1)

        # choose the min loss instance index
        self.instance_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.softmax_pred, labels=self.input_labels)
        self.min_loss_idx = tf.argmin(self.instance_loss, 0)

        # model loss
        self.model_loss = tf.reduce_mean(self.instance_loss)

        # optimizer
        if self.learning_rate:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.model_loss)
        else:
            self.optimizer = tf.train.AdamOptimizer().minimize(self.model_loss)

        # # saver
        # self.saver = tf.train.Saver(tf.all_variables())

    def fit(self, session, input_data, dropout_keep_rate):
        feed_dict = {self.input_words: input_data.word,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: dropout_keep_rate
                     }
        session.run(self.optimizer, feed_dict=feed_dict)
        model_loss = session.run(self.model_loss, feed_dict=feed_dict)
        return model_loss

    def predict(self, session, input_data):
        feed_dict = {self.input_words: input_data.word,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: 1}
        model_loss, label_pred = session.run([self.model_loss, self.class_label], feed_dict=feed_dict)
        return model_loss, label_pred


class BiGRU(object):
    def __init__(self, word_embedding, setting):
        # model name
        self.model_name = 'LSTM'

        # settings
        self.max_sentence_len = setting.sent_len
        self.hidden_size = setting.hidden_size
        self.class_num = setting.class_num
        self.pos_num = setting.pos_num
        self.pos_size = setting.pos_size
        self.learning_rate = setting.learning_rate

        # embedding matrix
        self.embed_matrix_word = tf.get_variable(
            'embed_matrix_word', word_embedding.shape,
            initializer=tf.constant_initializer(word_embedding)
        )
        self.embed_size_word = int(self.embed_matrix_word.get_shape()[1])
        self.embed_matrix_pos1 = tf.get_variable('embed_matrix_pos1', [self.pos_num, self.pos_size])
        self.embed_matrix_pos2 = tf.get_variable('embed_matrix_pos2', [self.pos_num, self.pos_size])

        # inputs
        self.input_words = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_words')
        self.input_labels = tf.placeholder(tf.int32, [None, self.class_num], name='labels')

        # position feature
        self.input_pos1 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos1')
        self.input_pos2 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos2')

        # dropout keep probability
        self.dropout_keep_rate = tf.placeholder(tf.float32, name="dropout_keep_rate")

        # embedded
        self.emb_word = tf.nn.embedding_lookup(self.embed_matrix_word, self.input_words)
        self.emb_pos1 = tf.nn.embedding_lookup(self.embed_matrix_pos1, self.input_pos1)
        self.emb_pos2 = tf.nn.embedding_lookup(self.embed_matrix_pos2, self.input_pos2)

        # concat embeddings
        self.emb_all = tf.concat([self.emb_word, self.emb_pos1, self.emb_pos2], 2)
        self.emb_all_us = tf.unstack(self.emb_all, num=self.max_sentence_len, axis = 1)

        # states and outputs
        with tf.name_scope('sentence_encoder'):
            # cell
            self.foward_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            self.backward_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            self.foward_cell = tf.nn.rnn_cell.DropoutWrapper(self.foward_cell, output_keep_prob=self.dropout_keep_rate)
            self.backward_cell = tf.nn.rnn_cell.DropoutWrapper(self.backward_cell, output_keep_prob=self.dropout_keep_rate)

            # rnn
            self.outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                self.foward_cell, self.backward_cell, self.emb_all_us, dtype=tf.float32
            )

        self.output_final = self.outputs[-1]

        # softmax
        with tf.name_scope('softmax'):
            self.softmax_w = tf.get_variable('softmax_W', [self.hidden_size * 2, self.class_num])
            self.softmax_b = tf.get_variable('softmax_b', [self.class_num])
            self.softmax_pred = tf.matmul(self.output_final, self.softmax_w) + self.softmax_b
            self.softmax_res = tf.nn.softmax(self.softmax_pred)

        # get max softmax predict result of each relation
        self.maxres_by_rel = tf.reduce_max(self.softmax_res, 0)

        # class label
        self.class_label = tf.argmax(self.softmax_res, 1)

        # choose the min loss instance index
        self.instance_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.softmax_pred, labels=self.input_labels)
        self.min_loss_idx = tf.argmin(self.instance_loss, 0)

        # model loss
        self.model_loss = tf.reduce_mean(self.instance_loss)

        # optimizer
        if self.learning_rate:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.model_loss)
        else:
            self.optimizer = tf.train.AdamOptimizer().minimize(self.model_loss)

        # # saver
        # self.saver = tf.train.Saver(tf.all_variables())

    def fit(self, session, input_data, dropout_keep_rate):
        feed_dict = {self.input_words: input_data.word,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: dropout_keep_rate
                     }
        session.run(self.optimizer, feed_dict=feed_dict)
        model_loss = session.run(self.model_loss, feed_dict=feed_dict)
        return model_loss

    def predict(self, session, input_data):
        feed_dict = {self.input_words: input_data.word,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: 1}
        model_loss, label_pred = session.run([self.model_loss, self.class_label], feed_dict=feed_dict)
        return model_loss, label_pred