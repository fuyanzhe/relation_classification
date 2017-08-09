# -*- encoding: utf-8 -*-
# Created by han on 17-7-11
import tensorflow as tf


class CNN(object):
    def __init__(self, is_training, word_embedding, setting):
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

        # max sentence length
        self.max_sentence_len = setting.sent_len

        # window size
        self.window_size = setting.window_size

        # filter number
        self.filter_sizes = setting.filter_sizes
        self.filter_num = setting.filter_num

        # number of classes
        self.class_num = setting.class_num

        # inputs
        self.input_words = tf.placeholder(tf.int32, [None, self.max_sentence_len, self.window_size], name='input_words')
        self.input_labels = tf.placeholder(tf.float32, [None, self.class_num], name='labels')

        # position feature
        self.input_pos1 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos1')
        self.input_pos2 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos2')

        # embedded
        self.emb_word = tf.nn.embedding_lookup(self.embed_matrix_word, self.input_words)
        self.emb_pos1 = tf.nn.embedding_lookup(self.embed_matrix_pos1, self.input_pos1)
        self.emb_pos1 = tf.nn.embedding_lookup(self.embed_matrix_pos2, self.input_pos2)

        # dropout keep probability
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # embedding of inputs
        # inputs size of words is changed to [batch_size, sentence_length, word_embedding_size * window_size]
        self.emb_inputs = tf.reshape(
            tf.nn.embedding_lookup(self.embed_matrix_word, self.input_words),
            [-1, self.max_sentence_len, self.window_size * self.embed_size_word]
        )

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

        # choose the min loss instance index
        self.instance_loss = tf.nn.softmax_cross_entropy_with_logits(self.softmax_pred, self.input_labels)
        self.min_loss_idx = tf.argmin(self.instance_loss, 0)

        # model loss
        self.model_loss = tf.reduce_mean(self.instance_loss)

        # result
        self.class_res = tf.argmax(self.softmax_res, 1)

        # saver
        self.saver = tf.train.Saver(tf.all_variables())

    def sentence_encoder(self):

        # convolution and max pooling
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # convolution layer
                filter_shape = [
                    filter_size, self.embed_size_word * self.window_size + 2 * self.embed_size_pos, 1, self.filter_num
                ]

                w = tf.get_variable(
                    'W', filter_shape,
                    initializer=tf.truncated_normal_initializer(stddev=0.1)
                )
                b = tf.get_variable(
                    'b', [self.filter_num],
                    initializer=tf.constant_initializer(0.1)
                )
                conv = tf.nn.conv2d(
                    self.emb_inputs_expanded, w, strides=[1, 1, 1, 1],
                    padding='VALID', name='conv'
                )

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
        h_pool = tf.concat(3, pooled_outputs)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Add dropout
        h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

        return h_drop

    def get_optimizer(self, learning_rate=None):
        # optimizer
        if learning_rate:
            return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.model_loss)
        else:
            return tf.train.AdamOptimizer().minimize(self.model_loss)


class RNNModel(object):
    def __init__(self, model_parameters):
        # model name
        self.model_name = 'RNN'

        # embedding matrix
        self.embed_matrix_words = tf.get_variable(
            'embed_matrix', model_parameters['embed_matrix_word'].shape,
            initializer=tf.constant_initializer(model_parameters['embed_matrix_word'])
        )
        self.embed_size_words = int(self.embed_matrix_words.get_shape()[1])

        # RNN length
        self.max_sentence_len = model_parameters['sent_len']

        # hidden size
        self.hidden_size = model_parameters['hidden_size']

        # number of classes
        self.class_num = model_parameters['class_num']

        # inputs
        self.input_words = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='sentences')
        self.input_labels = tf.placeholder(tf.float32, [None, self.class_num], name='labels')

        # sequence length of inputs
        self.seq_len = tf.placeholder(tf.int32, [None], name='len_sent')

        # dropout keep probability
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # embedding of inputs
        # inputs size of words is changed to [batch_size, sentence_length, word_embedding_size]
        self.emb_inputs = tf.nn.embedding_lookup(self.embed_matrix_words, self.input_words)
        # position feature
        self.pos_feature = model_parameters['position_feature']
        if self.pos_feature:
            self.embed_matrix_pos = tf.get_variable(
                'embed_matrix_pos', model_parameters['embed_matrix_pos'].shape,
                initializer=tf.constant_initializer(model_parameters['embed_matrix_pos'])
            )
            self.embed_size_pos = int(self.embed_matrix_pos.get_shape()[1])
            self.input_pos = tf.placeholder(tf.int32, [None, self.max_sentence_len, 2], name='input_positions')
            self.emb_pos = tf.reshape(
                tf.nn.embedding_lookup(self.embed_matrix_pos, self.input_pos),
                [-1, self.max_sentence_len, 2 * self.embed_size_pos]
            )
            self.emb_inputs = tf.concat(2, [self.emb_inputs, self.emb_pos])
        else:
            self.embed_size_pos = 0
        self.emb_inputs = tf.nn.dropout(self.emb_inputs, self.dropout_keep_prob)

        # cell
        self.cell_type = model_parameters['cell_type']
        self.cell = model_parameters['cell_type'][1](self.hidden_size + 2 * self.embed_size_pos)
        self.cell = DropoutWrapper(self.cell, self.dropout_keep_prob)

        # states and outputs
        self.outputs = self.sentence_encoder()

        # softmax
        self.softmax_w = tf.get_variable(
            'softmax_w', [self.hidden_size + 2 * self.embed_size_pos, self.class_num]
        )
        self.softmax_b = tf.get_variable('softmax_b', [self.class_num])
        self.softmax_pred = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b
        self.softmax_res = tf.nn.softmax(self.softmax_pred)

        # get max softmax predict result of each relation
        self.maxres_by_rel = tf.reduce_max(self.softmax_res, 0)

        # choose the min loss instance index
        self.instance_loss = tf.nn.softmax_cross_entropy_with_logits(self.softmax_pred, self.input_labels)
        self.min_loss_idx = tf.argmin(self.instance_loss, 0)

        # model loss
        self.model_loss = tf.reduce_mean(self.instance_loss)

        # result
        self.class_res = tf.argmax(self.softmax_res, 1)

        # saver
        self.saver = tf.train.Saver(tf.all_variables())

    def sentence_encoder(self):
        # word level RNN
        emb_list = tf.transpose(self.emb_inputs, [1, 0, 2])
        emb_list = tf.reshape(emb_list, [-1, self.embed_size_words + 2 * self.embed_size_pos])
        emb_list = tf.split(0, self.max_sentence_len, emb_list)

        # build RNN
        outputs, states = rnn(cell=self.cell, inputs=emb_list, sequence_length=self.seq_len, dtype=tf.float32)

        # get final output
        outputs = tf.pack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])
        batch_size = tf.shape(outputs)[0]
        index = tf.range(0, batch_size) * self.max_sentence_len + (self.seq_len - 1)
        outputs = tf.gather(tf.reshape(outputs, [-1, self.hidden_size + 2 * self.embed_size_pos]), index)

        return outputs

    def get_optimizer(self, learning_rate=None):
        # optimizerde
        if learning_rate:
            return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.model_loss)
        else:
            return tf.train.AdamOptimizer().minimize(self.model_loss)

class BiRNNModel(object):
    def __init__(self, model_parameters):
        # model name
        self.model_name = 'BiRNN'

        # embedding matrix
        self.embed_matrix_words = tf.get_variable(
            'embed_matrix', model_parameters['embed_matrix_word'].shape,
            initializer=tf.constant_initializer(model_parameters['embed_matrix_word'])
        )
        self.embed_size_words = int(self.embed_matrix_words.get_shape()[1])

        # RNN length
        self.max_sentence_len = model_parameters['sent_len']

        # hidden size
        self.hidden_size = model_parameters['hidden_size']

        # number of classes
        self.class_num = model_parameters['class_num']

        # inputs
        self.input_words = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='sentences')
        self.input_labels = tf.placeholder(tf.float32, [None, self.class_num], name='labels')

        # sequence length of inputs
        self.seq_len = tf.placeholder(tf.int32, [None], name='len_sent')

        # dropout keep probability
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # embedding of inputs
        # inputs size of words is changed to [batch_size, sentence_length, word_embedding_size]
        self.emb_inputs = tf.nn.embedding_lookup(self.embed_matrix_words, self.input_words)
        # position feature
        self.pos_feature = model_parameters['position_feature']
        if self.pos_feature:
            self.embed_matrix_pos = tf.get_variable(
                'embed_matrix_pos', model_parameters['embed_matrix_pos'].shape,
                initializer=tf.constant_initializer(model_parameters['embed_matrix_pos'])
            )
            self.embed_size_pos = int(self.embed_matrix_pos.get_shape()[1])
            self.input_pos = tf.placeholder(tf.int32, [None, self.max_sentence_len, 2], name='input_positions')
            self.emb_pos = tf.reshape(
                tf.nn.embedding_lookup(self.embed_matrix_pos, self.input_pos),
                [-1, self.max_sentence_len, 2 * self.embed_size_pos]
            )
            self.emb_inputs = tf.concat(2, [self.emb_inputs, self.emb_pos])
        else:
            self.embed_size_pos = 0
        self.emb_inputs = tf.nn.dropout(self.emb_inputs, self.dropout_keep_prob)

        # cell
        self.cell_type = model_parameters['cell_type']
        self.cell_fw = model_parameters['cell_type'][0](self.hidden_size + 2 * self.embed_size_pos)
        self.cell_bw = model_parameters['cell_type'][1](self.hidden_size + 2 * self.embed_size_pos)

        self.cell_fw = DropoutWrapper(self.cell_fw, self.dropout_keep_prob)
        self.cell_bw = DropoutWrapper(self.cell_bw, self.dropout_keep_prob)

        # states and outputs
        self.outputs = self.sentence_encoder()

        # softmax
        self.softmax_w = tf.get_variable(
            'softmax_w', [(self.hidden_size + 2 * self.embed_size_pos) * 2, self.class_num]
        )
        self.softmax_b = tf.get_variable('softmax_b', [self.class_num])
        self.softmax_pred = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b
        self.softmax_res = tf.nn.softmax(self.softmax_pred)

        # get max softmax predict result of each relation
        self.maxres_by_rel = tf.reduce_max(self.softmax_res, 0)

        # choose the min loss instance index
        self.instance_loss = tf.nn.softmax_cross_entropy_with_logits(self.softmax_pred, self.input_labels)
        self.min_loss_idx = tf.argmin(self.instance_loss, 0)

        # loss
        self.model_loss = tf.reduce_mean(self.instance_loss)

        # result
        self.class_res = tf.argmax(self.softmax_res, 1)

        # saver
        self.saver = tf.train.Saver(tf.all_variables())

    def sentence_encoder(self):
        # word level RNN
        emb_list = tf.transpose(self.emb_inputs, [1, 0, 2])
        emb_list = tf.reshape(emb_list, [-1, self.embed_size_words + 2 * self.embed_size_pos])
        emb_list = tf.split(0, self.max_sentence_len, emb_list)

        # build RNN
        outputs, states_fw, states_bw = bidirectional_rnn(
            cell_fw=self.cell_fw, cell_bw=self.cell_bw, inputs=emb_list, sequence_length=self.seq_len, dtype=tf.float32
        )

        # get final output
        outputs = tf.pack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])
        batch_size = tf.shape(outputs)[0]
        index = tf.range(0, batch_size) * self.max_sentence_len + (self.seq_len - 1)
        outputs = tf.gather(tf.reshape(outputs, [-1, (self.hidden_size + 2 * self.embed_size_pos) * 2]), index)

        return outputs

    def get_optimizer(self, learning_rate=None):
        # optimizer
        if learning_rate:
            return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.model_loss)
        else:
            return tf.train.AdamOptimizer().minimize(self.model_loss)
