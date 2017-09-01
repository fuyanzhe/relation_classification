# -*- encoding: utf-8 -*-
# Created by han on 17-7-11
import tensorflow as tf

rnn_cell = {
    'rnn': tf.nn.rnn_cell.BasicRNNCell,
    'lstm': tf.nn.rnn_cell.LSTMCell,
    'gru': tf.nn.rnn_cell.GRUCell
}


class Cnn(object):
    """
    Basic CNN model.
    """
    def __init__(self, x_embedding, setting):
        # model name
        self.model_name = 'Cnn'

        # embedding matrix
        self.embed_matrix_x = tf.get_variable(
            'embed_matrix_x', x_embedding.shape,
            initializer=tf.constant_initializer(x_embedding)
        )
        self.embed_size_x = int(self.embed_matrix_x.get_shape()[1])
        self.embed_matrix_pos1 = tf.get_variable('embed_matrix_pos1', [setting.pos_num, setting.pos_size])
        self.embed_matrix_pos2 = tf.get_variable('embed_matrix_pos2', [setting.pos_num, setting.pos_size])
        self.embed_size_pos = setting.pos_size

        # window size
        self.window_size = setting.win_size
        # max sentence length
        self.max_sentence_len = setting.sen_len

        # filter number
        self.filter_sizes = setting.filter_sizes
        self.filter_num = setting.filter_num

        # number of classes
        self.class_num = setting.class_num

        # inputs
        self.input_sen = tf.placeholder(tf.int32, [None, self.max_sentence_len, self.window_size], name='input_sen')
        self.input_labels = tf.placeholder(tf.int32, [None, self.class_num], name='labels')

        # position feature
        self.input_pos1 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos1')
        self.input_pos2 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos2')

        # dropout keep probability
        self.dropout_keep_rate = tf.placeholder(tf.float32, name="dropout_keep_rate")

        # learning rate
        self.learning_rate = setting.learning_rate

        # embedded
        self.emb_sen = tf.reshape(
            tf.nn.embedding_lookup(self.embed_matrix_x, self.input_sen),
            [-1, self.max_sentence_len, self.window_size * self.embed_size_x]
        )
        self.emb_pos1 = tf.nn.embedding_lookup(self.embed_matrix_pos1, self.input_pos1)
        self.emb_pos2 = tf.nn.embedding_lookup(self.embed_matrix_pos2, self.input_pos2)

        # concat embeddings
        self.emb_all = tf.concat([self.emb_sen, self.emb_pos1, self.emb_pos2], 2)
        self.emb_all_expanded = tf.expand_dims(self.emb_all, -1)

        # states and outputs
        with tf.name_scope('sentence_encoder'):
            self.outputs = self.sentence_encoder()

        # softmax
        with tf.name_scope('softmax'):
            # full connection layer before softmax
            self.softmax_w = tf.get_variable('softmax_W', [self.filter_num * len(self.filter_sizes), self.class_num])
            self.softmax_b = tf.get_variable('softmax_b', [self.class_num])
            self.softmax_pred = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b
            # self.softmax_pred = self.outputs
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
        self.saver = tf.train.Saver(tf.global_variables())

    def sentence_encoder(self):
        # convolution and max pooling
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # convolution layer
                filter_shape = [
                    filter_size, self.embed_size_x * self.window_size + 2 * self.embed_size_pos, 1, self.filter_num
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
        if self.window_size:
            input_x = input_data.win
        else:
            input_x = input_data.x
        feed_dict = {self.input_sen: input_x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: dropout_keep_rate
                     }
        session.run(self.optimizer, feed_dict=feed_dict)
        model_loss = session.run(self.model_loss, feed_dict=feed_dict)
        return model_loss

    def evaluate(self, session, input_data):
        if self.window_size:
            input_x = input_data.win
        else:
            input_x = input_data.x
        feed_dict = {self.input_sen: input_x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: 1}
        model_loss, label_pred, label_prob = session.run(
            [self.model_loss, self.class_label, self.softmax_res], feed_dict=feed_dict
        )
        return model_loss, label_pred, label_prob


class Rnn(object):
    """
    Basic Rnn model.
    """
    def __init__(self, x_embedding, setting):
        # model name
        self.model_name = 'Rnn'

        # settings
        self.cell_type = setting.cell
        self.max_sentence_len = setting.sen_len
        self.hidden_size = setting.hidden_size
        self.class_num = setting.class_num
        self.pos_num = setting.pos_num
        self.pos_size = setting.pos_size
        self.learning_rate = setting.learning_rate

        # embedding matrix
        self.embed_matrix_x = tf.get_variable(
            'embed_matrix_x', x_embedding.shape,
            initializer=tf.constant_initializer(x_embedding)
        )
        self.embed_size_x = int(self.embed_matrix_x.get_shape()[1])
        self.embed_matrix_pos1 = tf.get_variable('embed_matrix_pos1', [self.pos_num, self.pos_size])
        self.embed_matrix_pos2 = tf.get_variable('embed_matrix_pos2', [self.pos_num, self.pos_size])

        # inputs
        self.input_sen = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_sen')
        self.input_labels = tf.placeholder(tf.int32, [None, self.class_num], name='labels')

        # position feature
        self.input_pos1 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos1')
        self.input_pos2 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos2')

        # dropout keep probability
        self.dropout_keep_rate = tf.placeholder(tf.float32, name="dropout_keep_rate")

        # embedded
        self.emb_sen = tf.nn.embedding_lookup(self.embed_matrix_x, self.input_sen)
        self.emb_pos1 = tf.nn.embedding_lookup(self.embed_matrix_pos1, self.input_pos1)
        self.emb_pos2 = tf.nn.embedding_lookup(self.embed_matrix_pos2, self.input_pos2)

        # concat embeddings
        self.emb_all = tf.concat([self.emb_sen, self.emb_pos1, self.emb_pos2], 2)
        self.emb_all_us = tf.unstack(self.emb_all, num=self.max_sentence_len, axis=1)

        # states and outputs
        with tf.name_scope('sentence_encoder'):
            # cell
            self.rnn_cell = rnn_cell[self.cell_type](self.hidden_size)
            self.rnn_cell = tf.nn.rnn_cell.DropoutWrapper(self.rnn_cell, output_keep_prob=self.dropout_keep_rate)

            # rnn
            self.outputs, self.states = tf.contrib.rnn.static_rnn(
                self.rnn_cell, self.emb_all_us, dtype=tf.float32
            )

        if setting.hidden_select == 'last':
            self.output_final = self.outputs[-1]
        elif setting.hidden_select == 'avg':
            self.output_final = tf.reduce_mean(
                tf.reshape(tf.concat(self.outputs, 1), [-1, self.max_sentence_len, self.hidden_size]), axis=1
            )

        # softmax
        with tf.name_scope('softmax'):
            # full connection layer before softmax
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
        # self.saver = tf.train.Saver(tf.global_variables())

    def fit(self, session, input_data, dropout_keep_rate):
        feed_dict = {self.input_sen: input_data.x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: dropout_keep_rate
                     }
        session.run(self.optimizer, feed_dict=feed_dict)
        model_loss = session.run(self.model_loss, feed_dict=feed_dict)
        return model_loss

    def evaluate(self, session, input_data):
        feed_dict = {self.input_sen: input_data.x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: 1}
        model_loss, label_pred, label_prob = session.run(
            [self.model_loss, self.class_label, self.softmax_res], feed_dict=feed_dict
        )
        return model_loss, label_pred, label_prob


class BiRnn(object):
    """
    Bidirectional RNN model.
    """
    def __init__(self, x_embedding, setting):
        # model name
        self.model_name = 'BiRnn'

        # settings
        self.cell_type = setting.cell
        self.max_sentence_len = setting.sen_len
        self.hidden_size = setting.hidden_size
        self.class_num = setting.class_num
        self.pos_num = setting.pos_num
        self.pos_size = setting.pos_size
        self.learning_rate = setting.learning_rate

        # embedding matrix
        self.embed_matrix_x = tf.get_variable(
            'embed_matrix_x', x_embedding.shape,
            initializer=tf.constant_initializer(x_embedding)
        )
        self.embed_size_x = int(self.embed_matrix_x.get_shape()[1])
        self.embed_matrix_pos1 = tf.get_variable('embed_matrix_pos1', [self.pos_num, self.pos_size])
        self.embed_matrix_pos2 = tf.get_variable('embed_matrix_pos2', [self.pos_num, self.pos_size])

        # inputs
        self.input_sen = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_sen')
        self.input_labels = tf.placeholder(tf.int32, [None, self.class_num], name='labels')

        # position feature
        self.input_pos1 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos1')
        self.input_pos2 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos2')

        # dropout keep probability
        self.dropout_keep_rate = tf.placeholder(tf.float32, name="dropout_keep_rate")

        # embedded
        self.emb_sen = tf.nn.embedding_lookup(self.embed_matrix_x, self.input_sen)
        self.emb_pos1 = tf.nn.embedding_lookup(self.embed_matrix_pos1, self.input_pos1)
        self.emb_pos2 = tf.nn.embedding_lookup(self.embed_matrix_pos2, self.input_pos2)

        # concat embeddings
        self.emb_all = tf.concat([self.emb_sen, self.emb_pos1, self.emb_pos2], 2)
        self.emb_all_us = tf.unstack(self.emb_all, num=self.max_sentence_len, axis=1)

        # states and outputs
        with tf.name_scope('sentence_encoder'):
            # cell
            self.foward_cell = rnn_cell[self.cell_type](self.hidden_size)
            self.backward_cell = rnn_cell[self.cell_type](self.hidden_size)
            self.foward_cell = tf.nn.rnn_cell.DropoutWrapper(self.foward_cell, output_keep_prob=self.dropout_keep_rate)
            self.backward_cell = tf.nn.rnn_cell.DropoutWrapper(self.backward_cell, output_keep_prob=self.dropout_keep_rate)

            # rnn
            self.outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                self.foward_cell, self.backward_cell, self.emb_all_us, dtype=tf.float32
            )

            if setting.hidden_select == 'last':
                self.output_final = self.outputs[-1]
            elif setting.hidden_select == 'avg':
                self.output_final = tf.reduce_mean(
                    tf.reshape(tf.concat(self.outputs, 1), [-1, self.max_sentence_len, self.hidden_size]), axis=1
                )

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
        # self.saver = tf.train.Saver(tf.global_variables())

    def fit(self, session, input_data, dropout_keep_rate):
        feed_dict = {self.input_sen: input_data.x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: dropout_keep_rate
                     }
        session.run(self.optimizer, feed_dict=feed_dict)
        model_loss = session.run(self.model_loss, feed_dict=feed_dict)
        return model_loss

    def evaluate(self, session, input_data):
        feed_dict = {self.input_sen: input_data.x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: 1}
        model_loss, label_pred, label_prob = session.run(
            [self.model_loss, self.class_label, self.softmax_res], feed_dict=feed_dict
        )
        return model_loss, label_pred, label_prob


class BiRnn_Att(object):
    def __init__(self, x_embedding, setting):
        # model name
        self.model_name = 'BiRnn_Att'

        # settings
        self.cell_type = setting.cell
        self.max_sentence_len = setting.sen_len
        self.hidden_size = setting.hidden_size
        self.class_num = setting.class_num
        self.pos_num = setting.pos_num
        self.pos_size = setting.pos_size
        self.learning_rate = setting.learning_rate

        # embedding matrix
        self.embed_matrix_x = tf.get_variable(
            'embed_matrix_x', x_embedding.shape,
            initializer=tf.constant_initializer(x_embedding)
        )
        self.embed_size_x = int(self.embed_matrix_x.get_shape()[1])
        self.embed_matrix_pos1 = tf.get_variable('embed_matrix_pos1', [self.pos_num, self.pos_size])
        self.embed_matrix_pos2 = tf.get_variable('embed_matrix_pos2', [self.pos_num, self.pos_size])

        # inputs
        self.input_sen = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_sen')
        self.input_labels = tf.placeholder(tf.int32, [None, self.class_num], name='labels')

        # position feature
        self.input_pos1 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos1')
        self.input_pos2 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos2')

        # dropout keep probability
        self.dropout_keep_rate = tf.placeholder(tf.float32, name="dropout_keep_rate")

        # embedded
        self.emb_sen = tf.nn.embedding_lookup(self.embed_matrix_x, self.input_sen)
        self.emb_pos1 = tf.nn.embedding_lookup(self.embed_matrix_pos1, self.input_pos1)
        self.emb_pos2 = tf.nn.embedding_lookup(self.embed_matrix_pos2, self.input_pos2)

        # concat embeddings
        self.emb_all = tf.concat([self.emb_sen, self.emb_pos1, self.emb_pos2], 2)
        self.emb_all_us = tf.unstack(self.emb_all, num=self.max_sentence_len, axis=1)

        # states and outputs
        with tf.name_scope('sentence_encoder'):
            # cell
            self.foward_cell = rnn_cell[self.cell_type](self.hidden_size)
            self.backward_cell = rnn_cell[self.cell_type](self.hidden_size)
            self.foward_cell = tf.nn.rnn_cell.DropoutWrapper(self.foward_cell, output_keep_prob=self.dropout_keep_rate)
            self.backward_cell = tf.nn.rnn_cell.DropoutWrapper(self.backward_cell, output_keep_prob=self.dropout_keep_rate)

            # rnn
            with tf.name_scope('birnn'):
                self.outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                    self.foward_cell, self.backward_cell, self.emb_all_us, dtype=tf.float32
                )

            outputs_forward = [i[:, :self.hidden_size] for i in self.outputs]
            outputs_backward = [i[:, self.hidden_size:] for i in self.outputs]
            output_forward = tf.reshape(tf.concat(axis=1, values=outputs_forward), [-1, self.max_sentence_len, self.hidden_size])
            output_backward = tf.reshape(tf.concat(axis=1, values=outputs_backward), [-1, self.max_sentence_len, self.hidden_size])

            self.output_h = tf.add(output_forward, output_backward)

            # attention
            with tf.name_scope('attention'):
                self.attention_w = tf.get_variable('attention_omega', [self.hidden_size, 1])
                self.attention_A = tf.reshape(
                    tf.nn.softmax(
                        tf.reshape(
                            tf.matmul(
                                tf.reshape(tf.tanh(self.output_h), [-1, self.hidden_size]),
                                self.attention_w
                            ),
                            [-1, self.max_sentence_len]
                        )
                    ),
                    [-1, 1, self.max_sentence_len]
                )
                self.output_final = tf.reshape(tf.matmul(self.attention_A, self.output_h), [-1, self.hidden_size])

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
        self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                                              weights_list=tf.trainable_variables())
        self.model_loss = tf.reduce_mean(self.instance_loss) + self.l2_loss

        # optimizer
        if self.learning_rate:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.model_loss)
        else:
            self.optimizer = tf.train.AdamOptimizer().minimize(self.model_loss)

        # # saver
        # self.saver = tf.train.Saver(tf.global_variables())

    def fit(self, session, input_data, dropout_keep_rate):
        feed_dict = {self.input_sen: input_data.x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: dropout_keep_rate
                     }
        session.run(self.optimizer, feed_dict=feed_dict)
        model_loss = session.run(self.model_loss, feed_dict=feed_dict)
        return model_loss

    def evaluate(self, session, input_data):
        feed_dict = {self.input_sen: input_data.x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: 1}
        model_loss, label_pred, label_prob = session.run(
            [self.model_loss, self.class_label, self.softmax_res], feed_dict=feed_dict
        )
        return model_loss, label_pred, label_prob


class BiRnn_SelfAtt(object):
    def __init__(self, x_embedding, setting):
        # model name
        self.model_name = 'BiRnn_SelfAtt'

        # settings
        self.cell_type = setting.cell
        self.max_sentence_len = setting.sen_len
        self.hidden_size = setting.hidden_size
        self.class_num = setting.class_num
        self.pos_num = setting.pos_num
        self.pos_size = setting.pos_size
        self.learning_rate = setting.learning_rate

        # embedding matrix
        self.embed_matrix_x = tf.get_variable(
            'embed_matrix_x', x_embedding.shape,
            initializer=tf.constant_initializer(x_embedding)
        )
        self.embed_size_x = int(self.embed_matrix_x.get_shape()[1])
        self.embed_matrix_pos1 = tf.get_variable('embed_matrix_pos1', [self.pos_num, self.pos_size])
        self.embed_matrix_pos2 = tf.get_variable('embed_matrix_pos2', [self.pos_num, self.pos_size])

        # inputs
        self.input_sen = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_sen')
        self.input_labels = tf.placeholder(tf.int32, [None, self.class_num], name='labels')

        # position feature
        self.input_pos1 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos1')
        self.input_pos2 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos2')

        # dropout keep probability
        self.dropout_keep_rate = tf.placeholder(tf.float32, name="dropout_keep_rate")

        # embedded
        self.emb_sen = tf.nn.embedding_lookup(self.embed_matrix_x, self.input_sen)
        self.emb_pos1 = tf.nn.embedding_lookup(self.embed_matrix_pos1, self.input_pos1)
        self.emb_pos2 = tf.nn.embedding_lookup(self.embed_matrix_pos2, self.input_pos2)

        # concat embeddings
        self.emb_all = tf.concat([self.emb_sen, self.emb_pos1, self.emb_pos2], 2)
        self.emb_all_us = tf.unstack(self.emb_all, num=self.max_sentence_len, axis=1)

        # states and outputs
        with tf.name_scope('sentence_encoder'):
            # rnn
            with tf.name_scope('birnn'):
                # cell
                self.foward_cell = rnn_cell[self.cell_type](self.hidden_size)
                self.backward_cell = rnn_cell[self.cell_type](self.hidden_size)
                self.foward_cell = tf.nn.rnn_cell.DropoutWrapper(self.foward_cell,
                                                                 output_keep_prob=self.dropout_keep_rate)
                self.backward_cell = tf.nn.rnn_cell.DropoutWrapper(self.backward_cell,
                                                                   output_keep_prob=self.dropout_keep_rate)
                self.outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                    self.foward_cell, self.backward_cell, self.emb_all_us, dtype=tf.float32
                )
                self.output_h = tf.reshape(
                    tf.concat(self.outputs, 1), [-1, self.max_sentence_len, self.hidden_size * 2]
                )

            # attention
            with tf.name_scope('attention'):
                self.attention_Ws1 = tf.get_variable('attention_Ws1', [self.hidden_size * 2, setting.da])
                self.attention_Ws2 = tf.get_variable('attention_Ws2', [setting.da, setting.r])
                self.attention_A = tf.nn.softmax(
                    tf.transpose(
                        tf.reshape(
                            tf.matmul(
                                tf.tanh(
                                    tf.matmul(tf.reshape(self.output_h, [-1, self.hidden_size * 2]), self.attention_Ws1)
                                ),
                                self.attention_Ws2,
                            ),
                            [-1, self.max_sentence_len, setting.r]
                        ),
                        [0, 2, 1]
                    ),
                )

                self.M = tf.matmul(self.attention_A, self.output_h)

            with tf.name_scope('full_connection'):
                self.fc_w = tf.get_variable('fc_W', [self.hidden_size * 2, self.class_num])
                self.fc_b = tf.get_variable('fc_b', [self.class_num])
                self.sen_rep = tf.matmul(self.M, self.fc_w) + self.fc_b

        # softmax
        with tf.name_scope('softmax'):
            self.softmax_res = tf.nn.softmax(self.sen_rep)

        # get max softmax predict result of each relation
        self.maxres_by_rel = tf.reduce_max(self.softmax_res, 0)

        # class label
        self.class_label = tf.argmax(self.softmax_res, 1)

        # choose the min loss instance index
        self.instance_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.sen_rep, labels=self.input_labels)
        self.min_loss_idx = tf.argmin(self.instance_loss, 0)

        # Frobenius norm
        self.P_matrix = tf.matmul(
            self.attention_A,
            tf.transpose(self.attention_A, [0, 2, 1])
        ) - tf.eye(self.max_sentence_len, self.max_sentence_len)
        # self.P_loss = tf.pow(
        #     tf.norm(self.P_matrix, ord='fro', axis=1), [-2, -1]
        # )
        self.P_loss = tf.reduce_sum(self.P_matrix)

        # model loss
        self.model_loss = tf.reduce_mean(self.instance_loss) + 0.001 * self.P_loss

        # optimizer
        if self.learning_rate:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.model_loss)
        else:
            self.optimizer = tf.train.AdamOptimizer().minimize(self.model_loss)

        # # saver
        # self.saver = tf.train.Saver(tf.global_variables())

    def fit(self, session, input_data, dropout_keep_rate):
        feed_dict = {self.input_sen: input_data.word,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: dropout_keep_rate
                     }
        session.run(self.optimizer, feed_dict=feed_dict)
        model_loss = session.run(self.model_loss, feed_dict=feed_dict)
        return model_loss

    def evaluate(self, session, input_data):
        feed_dict = {self.input_sen: input_data.word,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: 1}
        model_loss, label_pred, label_prob = session.run(
            [self.model_loss, self.class_label, self.softmax_res], feed_dict=feed_dict
        )
        return model_loss, label_pred, label_prob


class BiRnn_Mi(object):
    def __init__(self, x_embedding, setting):
        # model name
        self.model_name = 'BiRnn_Mi'

        # settings
        self.cell_type = setting.cell
        self.max_sentence_len = setting.sen_len
        self.hidden_size = setting.hidden_size
        self.class_num = setting.class_num
        self.pos_num = setting.pos_num
        self.pos_size = setting.pos_size
        self.learning_rate = setting.learning_rate
        self.bag_num = setting.bag_num

        # embedding matrix
        self.embed_matrix_x = tf.get_variable(
            'embed_matrix_x', x_embedding.shape,
            initializer=tf.constant_initializer(x_embedding)
        )
        self.embed_size_x = int(self.embed_matrix_x.get_shape()[1])
        self.embed_matrix_pos1 = tf.get_variable('embed_matrix_pos1', [self.pos_num, self.pos_size])
        self.embed_matrix_pos2 = tf.get_variable('embed_matrix_pos2', [self.pos_num, self.pos_size])

        # shape of bags
        self.bag_shapes = tf.placeholder(tf.int32, [None], name='bag_shapes')
        self.instance_num = self.bag_shapes[-1]

        # inputs
        self.input_sen = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_sen')
        self.input_labels = tf.placeholder(tf.int32, [None, self.class_num], name='labels')

        # position feature
        self.input_pos1 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos1')
        self.input_pos2 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos2')

        # dropout keep probability
        self.dropout_keep_rate = tf.placeholder(tf.float32, name="dropout_keep_rate")

        # embedded
        self.emb_sen = tf.nn.embedding_lookup(self.embed_matrix_x, self.input_sen)
        self.emb_pos1 = tf.nn.embedding_lookup(self.embed_matrix_pos1, self.input_pos1)
        self.emb_pos2 = tf.nn.embedding_lookup(self.embed_matrix_pos2, self.input_pos2)

        # concat embeddings
        self.emb_all = tf.concat([self.emb_sen, self.emb_pos1, self.emb_pos2], 2)
        self.emb_all_us = tf.unstack(self.emb_all, num=self.max_sentence_len, axis=1)

        # states and outputs
        with tf.name_scope('sentence_encoder'):
            # cell
            self.foward_cell = rnn_cell[self.cell_type](self.hidden_size)
            self.backward_cell = rnn_cell[self.cell_type](self.hidden_size)
            self.foward_cell = tf.nn.rnn_cell.DropoutWrapper(
                self.foward_cell, output_keep_prob=self.dropout_keep_rate
            )
            self.backward_cell = tf.nn.rnn_cell.DropoutWrapper(
                self.backward_cell, output_keep_prob=self.dropout_keep_rate
            )

            # rnn
            self.outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                self.foward_cell, self.backward_cell, self.emb_all_us, dtype=tf.float32
            )
            self.sen_emb = self.outputs[-1]

        with tf.name_scope('sentence_attention'):
            # sentence-level attention layer
            sen_repre = []
            sen_alpha = []
            sen_s = []
            sen_out = []
            self.prob = []
            self.predictions = []
            self.loss = []
            self.accuracy = []
            self.total_loss = 0.0

            self.sen_a = tf.get_variable('attention_A', [self.hidden_size])
            self.sen_r = tf.get_variable('query_r', [self.hidden_size, 1])
            relation_embedding = tf.get_variable('relation_embedding', [self.class_num, self.hidden_size])
            sen_d = tf.get_variable('bias_d', [self.class_num])

            for i in range(self.bag_num):
                sen_repre.append(tf.tanh(self.sen_emb[self.bag_shapes[i]:self.bag_shapes[i + 1]]))
                bag_size = self.bag_shapes[i + 1] - self.bag_shapes[i]

                sen_alpha.append(
                    tf.reshape(
                        tf.nn.softmax(
                            tf.reshape(tf.matmul(tf.multiply(sen_repre[i], self.sen_a), self.sen_r), [bag_size])
                        ),
                        [1, bag_size]
                    )
                )

                sen_s.append(tf.reshape(tf.matmul(sen_alpha[i], sen_repre[i]), [self.hidden_size, 1]))
                sen_out.append(tf.add(tf.reshape(tf.matmul(relation_embedding, sen_s[i]), [self.class_num]), sen_d))

                self.prob.append(tf.nn.softmax(sen_out[i]))

                with tf.name_scope("output"):
                    self.predictions.append(tf.argmax(self.prob[i], 0, name="predictions"))

                with tf.name_scope("loss"):
                    self.loss.append(tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(logits=sen_out[i], labels=self.input_labels[i])))
                    if i == 0:
                        self.total_loss = self.loss[i]
                    else:
                        self.total_loss += self.loss[i]

                with tf.name_scope("accuracy"):
                    self.accuracy.append(
                        tf.reduce_mean(tf.cast(
                            tf.equal(self.predictions[i], tf.argmax(self.input_labels[i], 0)), "float"
                        ), name="accuracy"))

        # optimizer
        if self.learning_rate:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)
        else:
            self.optimizer = tf.train.AdamOptimizer().minimize(self.total_loss)

        # # saver
        # self.saver = tf.train.Saver(tf.global_variables())

    def fit(self, session, input_data, dropout_keep_rate):
        total_shape = [0]
        total_num = 0
        total_x = []
        total_pos1 = []
        total_pos2 = []
        for bag_idx in range(len(input_data.x)):
            total_num += len(input_data.word[bag_idx])
            total_shape.append(total_num)
            for sent in input_data.word[bag_idx]:
                total_x.append(sent)
            for pos1 in input_data.pos1[bag_idx]:
                total_pos1.append(pos1)
            for pos2 in input_data.pos2[bag_idx]:
                total_pos2.append(pos2)
        feed_dict = {
            self.bag_shapes: total_shape,
            self.input_sen: total_x,
            self.input_pos1: total_pos1,
            self.input_pos2: total_pos2,
            self.input_labels: input_data.y,
            self.dropout_keep_rate: dropout_keep_rate
        }
        session.run(self.optimizer, feed_dict=feed_dict)
        model_accuracy, model_loss = session.run([self.accuracy, self.total_loss], feed_dict=feed_dict)
        return model_accuracy, model_loss

    def evaluate(self, session, input_data):
        total_shape = [0]
        total_num = 0
        total_x = []
        total_pos1 = []
        total_pos2 = []
        for bag_idx in range(len(input_data.x)):
            total_num += len(input_data.word[bag_idx])
            total_shape.append(total_num)
            for sent in input_data.word[bag_idx]:
                total_x.append(sent)
            for pos1 in input_data.pos1[bag_idx]:
                total_pos1.append(pos1)
            for pos2 in input_data.pos2[bag_idx]:
                total_pos2.append(pos2)
        feed_dict = {
            self.bag_shapes: total_shape,
            self.input_sen: input_data.x,
            self.input_pos1: input_data.pos1,
            self.input_pos2: input_data.pos2,
            self.input_labels: input_data.y,
            self.dropout_keep_rate: 1
        }
        model_loss, label_pred, label_prob = session.run(
            [self.total_loss, self.predictions, self.prob], feed_dict=feed_dict
        )
        return model_loss, label_pred, label_prob
