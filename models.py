# -*- encoding: utf-8 -*-
# Created by han on 17-7-11
import numpy as np
import tensorflow as tf

rnn_cell = {
    'rnn': tf.nn.rnn_cell.BasicRNNCell,
    'lstm': tf.nn.rnn_cell.LSTMCell,
    'gru': tf.nn.rnn_cell.GRUCell
}

opt_method = {
    'sgd': tf.train.GradientDescentOptimizer,
    'adam': tf.train.AdamOptimizer,
    'adadelta': tf.train.AdadeltaOptimizer,
    'adagrad': tf.train.AdagradOptimizer
}


class Cnn(object):
    """
    Basic CNN model.
    """
    def __init__(self, x_embedding, setting):
        # model name
        self.model_name = 'Cnn'

        # max sentence length
        self.max_sentence_len = setting.sen_len

        # filter number
        self.filter_sizes = setting.filter_sizes
        self.filter_num = setting.filter_num

        # number of classes
        self.class_num = setting.class_num

        # learning rate
        self.learning_rate = setting.learning_rate

        with tf.name_scope('model_input'):
            # inputs
            self.input_sen = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_sen')
            self.input_labels = tf.placeholder(tf.int32, [None, self.class_num], name='labels')

            # position feature
            self.input_pos1 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos1')
            self.input_pos2 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos2')

            # dropout keep probability
            self.dropout_keep_rate = tf.placeholder(tf.float32, name="dropout_keep_rate")

        with tf.name_scope('embedding_layer'):
            # embedding matrix
            self.embed_matrix_x = tf.get_variable(
                'embed_matrix_x', x_embedding.shape,
                initializer=tf.constant_initializer(x_embedding)
            )
            self.embed_size_x = int(self.embed_matrix_x.get_shape()[1])
            self.embed_matrix_pos1 = tf.get_variable('embed_matrix_pos1', [setting.pos_num, setting.pos_size])
            self.embed_matrix_pos2 = tf.get_variable('embed_matrix_pos2', [setting.pos_num, setting.pos_size])
            self.embed_size_pos = setting.pos_size
            # embedded
            self.emb_sen = tf.nn.embedding_lookup(self.embed_matrix_x, self.input_sen)
            self.emb_pos1 = tf.nn.embedding_lookup(self.embed_matrix_pos1, self.input_pos1)
            self.emb_pos2 = tf.nn.embedding_lookup(self.embed_matrix_pos2, self.input_pos2)

            # concat embeddings
            self.emb_all = tf.concat([self.emb_sen, self.emb_pos1, self.emb_pos2], 2)
            self.emb_all_expanded = tf.expand_dims(self.emb_all, -1)

        with tf.name_scope('conv_maxpooling'):
            # convolution and max pooling
            pooled_outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope('conv-maxpool-{}'.format(filter_size)):
                    # convolution layer
                    filter_shape = [filter_size, self.embed_size_x + 2 * self.embed_size_pos, 1, self.filter_num]

                    w = tf.get_variable('W_{}'.format(filter_size),
                                        filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
                    b = tf.get_variable('b_{}'.format(filter_size),
                                        [self.filter_num], initializer=tf.constant_initializer(0.1))

                    tf.summary.histogram('W_{}'.format(filter_size), w)
                    tf.summary.histogram('b_{}'.format(filter_size), b)

                    conv = tf.nn.conv2d(self.emb_all_expanded, w, strides=[1, 1, 1, 1], padding='VALID', name='conv')

                    # Apply none linearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                    # Max pooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.max_sentence_len - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID', name='conv'
                    )
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = self.filter_num * len(self.filter_sizes)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

            # Add dropout
            self.outputs = tf.nn.dropout(h_pool_flat, self.dropout_keep_rate)

        # full connection layer
        with tf.name_scope('fc_layer'):
            # full connection layer before softmax
            self.fc_w = tf.get_variable('fc_W', [self.filter_num * len(self.filter_sizes), self.class_num])
            self.fc_b = tf.get_variable('fc_b', [self.class_num])

            tf.summary.histogram('fc_W', self.fc_w)
            tf.summary.histogram('fc_b', self.fc_b)

            self.fc_out = tf.matmul(self.outputs, self.fc_w) + self.fc_b

        # softmax
        with tf.name_scope('softmax'):
            self.softmax_res = tf.nn.softmax(self.fc_out)

        with tf.name_scope("accuracy"):
            # get max softmax predict result of each relation
            self.maxres_by_rel = tf.reduce_max(self.softmax_res, 0)

            # class label
            self.class_label = tf.argmax(self.softmax_res, 1)

            # accuracy
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.class_label, tf.argmax(self.input_labels, 1)), "float"), name="accuracy"
            )
            tf.summary.scalar('accuracy', self.accuracy)

        with tf.name_scope('model_predict'):
            # get max softmax predict result of each relation
            self.maxres_by_rel = tf.reduce_max(self.softmax_res, 0)

            # class label
            self.class_label = tf.argmax(self.softmax_res, 1)

        with tf.name_scope('model_loss'):
            # choose the min loss instance index
            self.instance_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc_out, labels=self.input_labels)

            # model loss
            self.model_loss = tf.reduce_mean(self.instance_loss)
            tf.summary.scalar('model_loss', self.model_loss)

        with tf.name_scope('optimizer'):
            # optimizer
            self.optimizer = opt_method[setting.optimizer](learning_rate=setting.learning_rate).minimize(
                self.model_loss)

        # tensor board summary
        self.merge_summary = tf.summary.merge_all()

    def fit(self, session, input_data, dropout_keep_rate):
        feed_dict = {self.input_sen: input_data.x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: dropout_keep_rate
                     }
        session.run(self.optimizer, feed_dict=feed_dict)
        summary, model_loss = session.run([self.merge_summary, self.model_loss], feed_dict=feed_dict)
        return summary, model_loss

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


class PCnn(object):
    """
    PCNN model.
    """
    def __init__(self, x_embedding, setting):
        # model name
        self.model_name = 'PCnn'

        # max sentence length
        self.max_sentence_len = setting.sen_len

        # filter number
        self.filter_sizes = setting.filter_sizes
        self.filter_num = setting.filter_num

        # number of classes
        self.class_num = setting.class_num

        # learning rate
        self.learning_rate = setting.learning_rate

        with tf.name_scope('model_input'):
            # inputs
            self.input_sen = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_sen')
            self.input_labels = tf.placeholder(tf.int32, [None, self.class_num], name='labels')

            # position feature
            self.input_pos1 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos1')
            self.input_pos2 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos2')

            # max pooling mask
            self.input_mask = tf.placeholder(tf.float32, [None, 3, self.max_sentence_len], name='input_mask')
            self.pcnn_mask = tf.expand_dims(tf.transpose(self.input_mask, [0, 2, 1]), axis=1)   # [batch, 1, L, 3]

            # dropout keep probability
            self.dropout_keep_rate = tf.placeholder(tf.float32, name="dropout_keep_rate")

        with tf.name_scope('embedding_layer'):
            # embedding matrix
            self.embed_matrix_x = tf.get_variable(
                'embed_matrix_x', x_embedding.shape,
                initializer=tf.constant_initializer(x_embedding)
            )
            self.embed_size_x = int(self.embed_matrix_x.get_shape()[1])
            self.embed_matrix_pos1 = tf.get_variable('embed_matrix_pos1', [setting.pos_num, setting.pos_size])
            self.embed_matrix_pos2 = tf.get_variable('embed_matrix_pos2', [setting.pos_num, setting.pos_size])
            self.embed_size_pos = setting.pos_size
            # embedded
            self.emb_sen = tf.nn.embedding_lookup(self.embed_matrix_x, self.input_sen)
            self.emb_pos1 = tf.nn.embedding_lookup(self.embed_matrix_pos1, self.input_pos1)
            self.emb_pos2 = tf.nn.embedding_lookup(self.embed_matrix_pos2, self.input_pos2)

            # concat embeddings
            self.emb_all = tf.concat([self.emb_sen, self.emb_pos1, self.emb_pos2], 2)
            self.conv_input = tf.expand_dims(self.emb_all, 1)   # [batch, 1, L, emb]

        with tf.name_scope('conv_maxpooling'):
            # convolution and max pooling
            pooled_outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope('conv-maxpool-{}'.format(filter_size)):
                    with tf.name_scope('conv-{}'.format(filter_size)):
                        # convolution layer
                        conv_out = tf.squeeze(
                            tf.nn.relu(
                                tf.layers.conv2d(
                                    self.conv_input, self.filter_num, [1, filter_size], padding='same',
                                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()
                                )
                            )
                        )
                        conv_out = tf.expand_dims(tf.transpose(conv_out, [0, 2, 1]), axis=-1)  # [batch, feat, L, 1]

                    with tf.name_scope('maxpool-{}'.format(filter_size)):
                        pcnn_pool = tf.reduce_max(conv_out * self.pcnn_mask, axis=2)  # [batch, feat, 3]
                        pcnn_pool = tf.reshape(pcnn_pool, [-1, self.filter_num * 3])
                        pooled_outputs.append(pcnn_pool)

            # Combine all the pooled features
            feature_size = self.filter_num * 3 * len(self.filter_sizes)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, feature_size])

            # Add dropout
            self.outputs = tf.nn.dropout(h_pool_flat, self.dropout_keep_rate)

        # full connection layer
        with tf.name_scope('fc_layer'):
            # full connection layer before softmax
            self.fc_w = tf.get_variable('fc_W', [feature_size, self.class_num])
            self.fc_b = tf.get_variable('fc_b', [self.class_num])

            tf.summary.histogram('fc_W', self.fc_w)
            tf.summary.histogram('fc_b', self.fc_b)

            self.fc_out = tf.matmul(self.outputs, self.fc_w) + self.fc_b

        # softmax
        with tf.name_scope('softmax'):
            self.softmax_res = tf.nn.softmax(self.fc_out)

        with tf.name_scope("accuracy"):
            # get max softmax predict result of each relation
            self.maxres_by_rel = tf.reduce_max(self.softmax_res, 0)

            # class label
            self.class_label = tf.argmax(self.softmax_res, 1)

            # accuracy
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.class_label, tf.argmax(self.input_labels, 1)), "float"), name="accuracy"
            )
            tf.summary.scalar('accuracy', self.accuracy)

        with tf.name_scope('model_predict'):
            # get max softmax predict result of each relation
            self.maxres_by_rel = tf.reduce_max(self.softmax_res, 0)

            # class label
            self.class_label = tf.argmax(self.softmax_res, 1)

        with tf.name_scope('model_loss'):
            # choose the min loss instance index
            self.instance_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc_out, labels=self.input_labels)

            # model loss
            self.model_loss = tf.reduce_mean(self.instance_loss)
            tf.summary.scalar('model_loss', self.model_loss)

        with tf.name_scope('optimizer'):
            # optimizer
            self.optimizer = opt_method[setting.optimizer](learning_rate=setting.learning_rate).minimize(
                self.model_loss)

        # tensor board summary
        self.merge_summary = tf.summary.merge_all()

    def fit(self, session, input_data, dropout_keep_rate):
        feed_dict = {self.input_sen: input_data.x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_mask: input_data.mask,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: dropout_keep_rate
                     }
        session.run(self.optimizer, feed_dict=feed_dict)
        summary, model_loss = session.run([self.merge_summary, self.model_loss], feed_dict=feed_dict)
        return summary, model_loss

    def evaluate(self, session, input_data):
        feed_dict = {self.input_sen: input_data.x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_mask: input_data.mask,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: 1}
        model_loss, label_pred, label_prob = session.run(
            [self.model_loss, self.class_label, self.softmax_res], feed_dict=feed_dict
        )
        return model_loss, label_pred, label_prob


class Cnn_Deep(object):
    """
    Multi-layer CNN model.
    """
    def __init__(self, x_embedding, setting):
        # model name
        self.model_name = 'Cnn_Deep'

        with tf.name_scope('model_input'):
            # max sentence length
            self.max_sentence_len = setting.sen_len

            # filter number
            self.filter_sizes = setting.filter_sizes
            self.filter_num = setting.filter_num

            # max pooling settings
            self.max_pool_sizes = setting.max_pool_sizes

            # full connecting settings
            self.fc_sizes = setting.fc_sizes

            # number of classes
            self.class_num = setting.class_num

            # inputs
            self.input_sen = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_sen')
            self.input_labels = tf.placeholder(tf.int32, [None, self.class_num], name='labels')

            # position feature
            self.input_pos1 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos1')
            self.input_pos2 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos2')

            # dropout
            self.dropout_mask = setting.dropout_mask
            self.dropout_keep_rate = tf.placeholder(tf.float32, name="dropout_keep_rate")

            # embedding matrix
            self.embed_matrix_x = tf.get_variable(
                'embed_matrix_x', x_embedding.shape,
                initializer=tf.constant_initializer(x_embedding)
            )
            self.embed_size_x = int(self.embed_matrix_x.get_shape()[1])
            self.embed_matrix_pos1 = tf.get_variable('embed_matrix_pos1', [setting.pos_num, setting.pos_size])
            self.embed_matrix_pos2 = tf.get_variable('embed_matrix_pos2', [setting.pos_num, setting.pos_size])
            self.embed_size_pos = setting.pos_size

        with tf.name_scope('embedding_layer'):
            # embedded
            self.emb_sen = tf.reshape(
                tf.nn.embedding_lookup(self.embed_matrix_x, self.input_sen),
                [-1, self.max_sentence_len, self.embed_size_x]
            )
            self.emb_pos1 = tf.nn.embedding_lookup(self.embed_matrix_pos1, self.input_pos1)
            self.emb_pos2 = tf.nn.embedding_lookup(self.embed_matrix_pos2, self.input_pos2)

            # concat embeddings
            self.emb_all = tf.concat([self.emb_sen, self.emb_pos1, self.emb_pos2], 2)
            self.emb_all_expanded = tf.expand_dims(self.emb_all, -1)

        with tf.name_scope('conv_maxpooling'):
            # input shape of next conv layer
            nl_input = self.emb_all_expanded

            # conv and max pool
            for i in range(len(self.filter_sizes)):
                filter_size = self.filter_sizes[i]
                filter_num = self.filter_num[i]
                # convolution and max pooling
                with tf.name_scope('conv_{}'.format(i)):
                    # convolution layer
                    filter_shape = [filter_size, nl_input.shape[2], 1, filter_num]

                    w = tf.get_variable('W_conv_{}'.format(i),
                                        filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.05))
                    b = tf.get_variable('b_conv_{}'.format(i),
                                        [filter_num], initializer=tf.truncated_normal_initializer(stddev=0.05))

                    tf.summary.histogram('W_conv_{}'.format(i), w)
                    tf.summary.histogram('b_conv_{}'.format(i), b)

                    conv = tf.nn.conv2d(nl_input, w, strides=[1, 1, 1, 1], padding='VALID', name='conv')

                    # Apply none linearity
                    nl_input = tf.transpose(tf.nn.relu(tf.nn.bias_add(conv, b), name='relu'), [0, 1, 3, 2])

                if self.max_pool_sizes[i] != 0:
                    with tf.name_scope('max_pool_{}'.format(i)):
                        max_pool_size = self.max_pool_sizes[i]
                        # Max pooling over the outputs
                        nl_input = tf.nn.max_pool(
                            nl_input,
                            ksize=[1, max_pool_size[0], max_pool_size[1], 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID', name='max_pool_{}'.format(i)
                        )

                if self.dropout_mask[i] != 0:
                    self.nl_input = tf.nn.dropout(nl_input, self.dropout_keep_rate)

        # full connection layer
        with tf.name_scope('fc_layers'):
            nl_input = tf.reshape(nl_input, [-1, int(nl_input.shape[1]) * int(nl_input.shape[2])])
            for i in range(len(self.fc_sizes)):
                with tf.name_scope('fc_layer_{}'.format(i)):
                    # full connection layer before softmax
                    fc_w = tf.get_variable('fc_W_{}'.format(i), [nl_input.shape[-1], self.fc_sizes[i]],
                                           initializer=tf.truncated_normal_initializer(stddev=0.05))
                    fc_b = tf.get_variable('fc_b_{}'.format(i), [self.fc_sizes[i]],
                                           initializer=tf.truncated_normal_initializer(stddev=0.05))

                    tf.summary.histogram('fc_W_{}'.format(i), fc_w)
                    tf.summary.histogram('fc_b_{}'.format(i), fc_b)

                    nl_input = tf.matmul(nl_input, fc_w) + fc_b

                    if self.dropout_mask[i - len(self.filter_sizes)] != 0:
                        self.nl_input = tf.nn.dropout(nl_input, self.dropout_keep_rate)

            self.fc_out = nl_input

        # softmax
        with tf.name_scope('softmax'):
            self.softmax_res = tf.nn.softmax(self.fc_out)

        with tf.name_scope("accuracy"):
            # class label
            self.class_label = tf.argmax(self.softmax_res, 1)

            # accuracy
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.class_label, tf.argmax(self.input_labels, 1)), "float"), name="accuracy"
            )
            tf.summary.scalar('accuracy', self.accuracy)

        with tf.name_scope('model_predict'):
            # class label
            self.class_label = tf.argmax(self.softmax_res, 1)

        with tf.name_scope('model_loss'):
            # choose the min loss instance index
            self.instance_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc_out, labels=self.input_labels)

            # model loss
            self.model_loss = tf.reduce_mean(self.instance_loss)
            tf.summary.scalar('model_loss', self.model_loss)

        with tf.name_scope('optimizer'):
            # optimizer
            self.optimizer = opt_method[setting.optimizer](learning_rate=setting.learning_rate).minimize(
                self.model_loss)

        # tensor board summary
        self.merge_summary = tf.summary.merge_all()

    def fit(self, session, input_data, dropout_keep_rate):
        input_x = input_data.x
        feed_dict = {self.input_sen: input_x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: dropout_keep_rate
                     }
        session.run(self.optimizer, feed_dict=feed_dict)
        summary, model_loss = session.run([self.merge_summary, self.model_loss], feed_dict=feed_dict)
        return summary, model_loss

    def evaluate(self, session, input_data):
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

        with tf.name_scope('model_input'):
            # inputs
            self.input_sen = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_sen')
            self.input_labels = tf.placeholder(tf.int32, [None, self.class_num], name='labels')

            # position feature
            self.input_pos1 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos1')
            self.input_pos2 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos2')

            # dropout keep probability
            self.dropout_keep_rate = tf.placeholder(tf.float32, name="dropout_keep_rate")

        with tf.name_scope('embedding_layer'):
            # embedding matrix
            self.embed_matrix_x = tf.get_variable(
                'embed_matrix_x', x_embedding.shape,
                initializer=tf.constant_initializer(x_embedding)
            )
            self.embed_size_x = int(self.embed_matrix_x.get_shape()[1])
            self.embed_matrix_pos1 = tf.get_variable('embed_matrix_pos1', [self.pos_num, self.pos_size])
            self.embed_matrix_pos2 = tf.get_variable('embed_matrix_pos2', [self.pos_num, self.pos_size])

            # embedded
            self.emb_sen = tf.nn.embedding_lookup(self.embed_matrix_x, self.input_sen)
            self.emb_pos1 = tf.nn.embedding_lookup(self.embed_matrix_pos1, self.input_pos1)
            self.emb_pos2 = tf.nn.embedding_lookup(self.embed_matrix_pos2, self.input_pos2)

            # concat embeddings
            self.emb_all = tf.concat([self.emb_sen, self.emb_pos1, self.emb_pos2], 2)
            self.emb_all_us = tf.unstack(self.emb_all, num=self.max_sentence_len, axis=1)

        # states and outputs
        with tf.name_scope('rnn_layer'):
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

        with tf.name_scope('fc_layer'):
            # full connection layer before softmax
            self.fc_w = tf.get_variable('fc_W', [self.hidden_size, self.class_num])
            self.fc_b = tf.get_variable('fc_b', [self.class_num])
            self.fc_output = tf.matmul(self.output_final, self.fc_w) + self.fc_b

            tf.summary.histogram('fc_w', self.fc_w)
            tf.summary.histogram('fc_b', self.fc_b)

        with tf.name_scope('softmax'):
            self.softmax_res = tf.nn.softmax(self.fc_output)

        with tf.name_scope('model_predict'):
            # class label
            self.class_label = tf.argmax(self.softmax_res, 1)

        with tf.name_scope('model_loss'):
            self.loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.fc_output, labels=self.input_labels
            )
            # model loss
            self.model_loss = tf.reduce_mean(self.loss)

            tf.summary.scalar('model_loss', self.model_loss)

        with tf.name_scope('optimizer'):
            # optimizer
            self.optimizer = opt_method[setting.optimizer](learning_rate=setting.learning_rate).minimize(
                self.model_loss)

        # tensor board summary
        self.merge_summary = tf.summary.merge_all()

    def fit(self, session, input_data, dropout_keep_rate):
        feed_dict = {self.input_sen: input_data.x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: dropout_keep_rate
                     }
        session.run(self.optimizer, feed_dict=feed_dict)
        summary, model_loss = session.run([self.merge_summary, self.model_loss], feed_dict=feed_dict)
        return summary, model_loss

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

        with tf.name_scope('model_input'):
            # inputs
            self.input_sen = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_sen')
            self.input_labels = tf.placeholder(tf.int32, [None, self.class_num], name='labels')

            # position feature
            self.input_pos1 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos1')
            self.input_pos2 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos2')

            # dropout keep probability
            self.dropout_keep_rate = tf.placeholder(tf.float32, name="dropout_keep_rate")

        with tf.name_scope('embedding_layer'):
            # embedding matrix
            self.embed_matrix_x = tf.get_variable(
                'embed_matrix_x', x_embedding.shape,
                initializer=tf.constant_initializer(x_embedding)
            )
            self.embed_size_x = int(self.embed_matrix_x.get_shape()[1])
            self.embed_matrix_pos1 = tf.get_variable('embed_matrix_pos1', [self.pos_num, self.pos_size])
            self.embed_matrix_pos2 = tf.get_variable('embed_matrix_pos2', [self.pos_num, self.pos_size])

            # embedded
            self.emb_sen = tf.nn.embedding_lookup(self.embed_matrix_x, self.input_sen)
            self.emb_pos1 = tf.nn.embedding_lookup(self.embed_matrix_pos1, self.input_pos1)
            self.emb_pos2 = tf.nn.embedding_lookup(self.embed_matrix_pos2, self.input_pos2)

            # concat embeddings
            self.emb_all = tf.concat([self.emb_sen, self.emb_pos1, self.emb_pos2], 2)
            self.emb_all_us = tf.unstack(self.emb_all, num=self.max_sentence_len, axis=1)

        # states and outputs
        with tf.name_scope('rnn_layer'):
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
            self.rnn_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                self.foward_cell, self.backward_cell, self.emb_all_us, dtype=tf.float32
            )

            if setting.hidden_select == 'last':
                self.rnn_output = self.rnn_outputs[-1]
            elif setting.hidden_select == 'avg':
                self.rnn_output = tf.reduce_mean(
                    tf.reshape(tf.concat(self.rnn_outputs, 1), [-1, self.max_sentence_len, self.hidden_size * 2]), axis=1
                )

        with tf.name_scope('fc_layer'):
            self.fc_w = tf.get_variable('fc_W', [self.hidden_size * 2, self.class_num])
            self.fc_b = tf.get_variable('fc_b', [self.class_num])
            self.fc_output = tf.matmul(self.rnn_output, self.fc_w) + self.fc_b

            tf.summary.histogram('fc_w', self.fc_w)
            tf.summary.histogram('fc_b', self.fc_b)

        with tf.name_scope('softmax'):
            self.softmax_output = tf.nn.softmax(self.fc_output)

        with tf.name_scope('model_predict'):
            # class label
            self.class_label = tf.argmax(self.softmax_output, 1)

        with tf.name_scope('model_loss'):
            # choose the min loss instance index
            self.instance_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc_output, labels=self.input_labels)
            # model loss
            self.model_loss = tf.reduce_mean(self.instance_loss)

            tf.summary.scalar('model_loss', self.model_loss)

        with tf.name_scope('optimizer'):
            # optimizer
            self.optimizer = opt_method[setting.optimizer](learning_rate=setting.learning_rate).minimize(
                self.model_loss)

        # tensor board summary
        self.merge_summary = tf.summary.merge_all()

    def fit(self, session, input_data, dropout_keep_rate):
        feed_dict = {self.input_sen: input_data.x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: dropout_keep_rate
                     }
        session.run(self.optimizer, feed_dict=feed_dict)
        summary, model_loss = session.run([self.merge_summary, self.model_loss], feed_dict=feed_dict)
        return summary, model_loss

    def evaluate(self, session, input_data):
        feed_dict = {self.input_sen: input_data.x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: 1}
        model_loss, label_pred, label_prob = session.run(
            [self.model_loss, self.class_label, self.softmax_output], feed_dict=feed_dict
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

        with tf.name_scope('model_input'):
            # inputs
            self.input_sen = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_sen')
            self.input_labels = tf.placeholder(tf.int32, [None, self.class_num], name='labels')

            # position feature
            self.input_pos1 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos1')
            self.input_pos2 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos2')

            # dropout keep probability
            self.dropout_keep_rate = tf.placeholder(tf.float32, name="dropout_keep_rate")

        with tf.name_scope('embedding_layer'):
            # embedding matrix
            self.embed_matrix_x = tf.get_variable(
                'embed_matrix_x', x_embedding.shape,
                initializer=tf.constant_initializer(x_embedding)
            )
            self.embed_size_x = int(self.embed_matrix_x.get_shape()[1])
            self.embed_matrix_pos1 = tf.get_variable('embed_matrix_pos1', [self.pos_num, self.pos_size])
            self.embed_matrix_pos2 = tf.get_variable('embed_matrix_pos2', [self.pos_num, self.pos_size])

            # embedded
            self.emb_sen = tf.nn.embedding_lookup(self.embed_matrix_x, self.input_sen)
            self.emb_pos1 = tf.nn.embedding_lookup(self.embed_matrix_pos1, self.input_pos1)
            self.emb_pos2 = tf.nn.embedding_lookup(self.embed_matrix_pos2, self.input_pos2)

            # concat embeddings
            self.emb_all = tf.concat([self.emb_sen, self.emb_pos1, self.emb_pos2], 2)
            self.emb_all_us = tf.unstack(self.emb_all, num=self.max_sentence_len, axis=1)

        # states and outputs
        with tf.name_scope('rnn_layer'):
            with tf.name_scope('birnn'):
                self.foward_cell = rnn_cell[self.cell_type](self.hidden_size)
                self.backward_cell = rnn_cell[self.cell_type](self.hidden_size)
                self.foward_cell = tf.nn.rnn_cell.DropoutWrapper(self.foward_cell, output_keep_prob=self.dropout_keep_rate)
                self.backward_cell = tf.nn.rnn_cell.DropoutWrapper(self.backward_cell, output_keep_prob=self.dropout_keep_rate)

                rnn_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                    self.foward_cell, self.backward_cell, self.emb_all_us, dtype=tf.float32
                )
                outputs_forward = [i[:, :self.hidden_size] for i in rnn_outputs]
                outputs_backward = [i[:, self.hidden_size:] for i in rnn_outputs]
                output_forward = tf.reshape(tf.concat(axis=1, values=outputs_forward),
                                            [-1, self.max_sentence_len, self.hidden_size])
                output_backward = tf.reshape(tf.concat(axis=1, values=outputs_backward),
                                             [-1, self.max_sentence_len, self.hidden_size])

                self.rnn_outputs = tf.add(output_forward, output_backward)

            # attention
            with tf.name_scope('attention'):
                self.attention_w = tf.get_variable('attention_omega', [self.hidden_size, 1])
                self.attention_A = tf.reshape(
                    tf.nn.softmax(
                        tf.reshape(
                            tf.matmul(
                                tf.reshape(tf.tanh(self.rnn_outputs), [-1, self.hidden_size]),
                                self.attention_w
                            ),
                            [-1, self.max_sentence_len]
                        )
                    ),
                    [-1, 1, self.max_sentence_len]
                )
                self.rnn_output = tf.reshape(tf.matmul(self.attention_A, self.rnn_outputs), [-1, self.hidden_size])

        with tf.name_scope('fc_layer'):
            self.fc_w = tf.get_variable('fc_W', [self.hidden_size, self.class_num])
            self.fc_b = tf.get_variable('fc_b', [self.class_num])
            self.fc_output = tf.matmul(self.rnn_output, self.fc_w) + self.fc_b

            tf.summary.histogram('fc_w', self.fc_w)
            tf.summary.histogram('fc_b', self.fc_b)

        with tf.name_scope('softmax'):
            self.softmax_output = tf.nn.softmax(self.fc_output)

        with tf.name_scope('model_predict'):
            self.class_label = tf.argmax(self.softmax_output, 1)

        with tf.name_scope('model_loss'):
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc_output,
                                                                labels=self.input_labels)
            self.l2_regular = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                                                     weights_list=tf.trainable_variables())
            self.model_loss = tf.reduce_mean(self.loss) + self.l2_regular

            tf.summary.scalar('model_loss', self.model_loss)

        with tf.name_scope('optimizer'):
            # optimizer
            self.optimizer = opt_method[setting.optimizer](learning_rate=setting.learning_rate).minimize(
                self.model_loss)

        # tensor board summary
        self.merge_summary = tf.summary.merge_all()

    def fit(self, session, input_data, dropout_keep_rate):
        feed_dict = {self.input_sen: input_data.x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: dropout_keep_rate
                     }
        session.run(self.optimizer, feed_dict=feed_dict)
        summary, model_loss = session.run([self.merge_summary, self.model_loss], feed_dict=feed_dict)
        return summary, model_loss

    def evaluate(self, session, input_data):
        feed_dict = {self.input_sen: input_data.x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: 1}
        model_loss, label_pred, label_prob = session.run(
            [self.model_loss, self.class_label, self.softmax_output], feed_dict=feed_dict
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

        with tf.name_scope('model_input'):
            # inputs
            self.input_sen = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_sen')
            self.input_labels = tf.placeholder(tf.int32, [None, self.class_num], name='labels')

            # position feature
            self.input_pos1 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos1')
            self.input_pos2 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos2')

            # dropout keep probability
            self.dropout_keep_rate = tf.placeholder(tf.float32, name="dropout_keep_rate")

        with tf.name_scope('embedding_layer'):
            # embedding matrix
            self.embed_matrix_x = tf.get_variable(
                'embed_matrix_x', x_embedding.shape,
                initializer=tf.constant_initializer(x_embedding)
            )
            self.embed_size_x = int(self.embed_matrix_x.get_shape()[1])
            self.embed_matrix_pos1 = tf.get_variable('embed_matrix_pos1', [self.pos_num, self.pos_size])
            self.embed_matrix_pos2 = tf.get_variable('embed_matrix_pos2', [self.pos_num, self.pos_size])

            # embedded
            self.emb_sen = tf.nn.embedding_lookup(self.embed_matrix_x, self.input_sen)
            self.emb_pos1 = tf.nn.embedding_lookup(self.embed_matrix_pos1, self.input_pos1)
            self.emb_pos2 = tf.nn.embedding_lookup(self.embed_matrix_pos2, self.input_pos2)

            # concat embeddings
            self.emb_all = tf.concat([self.emb_sen, self.emb_pos1, self.emb_pos2], 2)
            self.emb_all_us = tf.unstack(self.emb_all, num=self.max_sentence_len, axis=1)

        # states and outputs
        with tf.name_scope('rnn_layer'):
            with tf.name_scope('birnn'):
                # cell
                self.foward_cell = rnn_cell[self.cell_type](self.hidden_size)
                self.backward_cell = rnn_cell[self.cell_type](self.hidden_size)
                self.foward_cell = tf.nn.rnn_cell.DropoutWrapper(self.foward_cell,
                                                                 output_keep_prob=self.dropout_keep_rate)
                self.backward_cell = tf.nn.rnn_cell.DropoutWrapper(self.backward_cell,
                                                                   output_keep_prob=self.dropout_keep_rate)
                self.rnn_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                    self.foward_cell, self.backward_cell, self.emb_all_us, dtype=tf.float32
                )
                self.output_h = tf.reshape(
                    tf.concat(self.rnn_outputs, 1), [-1, self.max_sentence_len, self.hidden_size * 2]
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

                tf.summary.histogram('Ws1', self.attention_Ws1)
                tf.summary.histogram('Ws2', self.attention_Ws2)

        with tf.name_scope('full_connection'):
            self.vec_M = tf.reshape(self.M, [-1,  self.hidden_size * 2 * setting.r])
            self.fc_w = tf.get_variable('fc_W', [self.hidden_size * 2 * setting.r, self.class_num])
            self.fc_b = tf.get_variable('fc_b', [self.class_num])
            self.sen_rep = tf.matmul(self.vec_M, self.fc_w) + self.fc_b

            tf.summary.histogram('vec_M', self.vec_M)
            tf.summary.histogram('fc_w', self.fc_w)
            tf.summary.histogram('fc_b', self.fc_b)

        with tf.name_scope('softmax'):
            self.softmax_output = tf.nn.softmax(self.sen_rep)

        with tf.name_scope('model_predict'):
            self.class_label = tf.argmax(self.softmax_output, 1)

        with tf.name_scope('model_loss'):
            # choose the min loss instance index
            self.instance_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.sen_rep, labels=self.input_labels)

            # Frobenius norm
            self.P_att_matrix = tf.matmul(self.attention_A, tf.transpose(self.attention_A, [0, 2, 1])) - tf.eye(
                setting.r, setting.r)
            self.P_att = tf.pow(tf.norm(self.P_att_matrix), 2)

            # model loss
            self.model_loss = tf.reduce_mean(self.instance_loss) + 0.0001 * self.P_att

        with tf.name_scope('optimizer'):
            # optimizer
            self.optimizer = opt_method[setting.optimizer](learning_rate=setting.learning_rate).minimize(
                self.model_loss)

        # tensor board summary
        self.merge_summary = tf.summary.merge_all()

    def fit(self, session, input_data, dropout_keep_rate):
        feed_dict = {self.input_sen: input_data.x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: dropout_keep_rate
                     }
        session.run(self.optimizer, feed_dict=feed_dict)
        summary, model_loss = session.run([self.merge_summary, self.model_loss], feed_dict=feed_dict)
        return summary, model_loss

    def evaluate(self, session, input_data):
        feed_dict = {self.input_sen: input_data.x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: 1}
        model_loss, label_pred, label_prob = session.run(
            [self.model_loss, self.class_label, self.softmax_output], feed_dict=feed_dict
        )
        return model_loss, label_pred, label_prob


class BiRnn_Res(object):
    """
    Bidirectional RNN model.
    """
    def __init__(self, x_embedding, setting):
        # model name
        self.model_name = 'BiRnn_Res'

        # settings
        self.cell_type = setting.cell
        self.max_sentence_len = setting.sen_len
        self.hidden_size = setting.hidden_size
        self.class_num = setting.class_num
        self.pos_num = setting.pos_num
        self.pos_size = setting.pos_size
        self.learning_rate = setting.learning_rate

        with tf.name_scope('model_input'):
            # inputs
            self.input_sen = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_sen')
            self.input_labels = tf.placeholder(tf.int32, [None, self.class_num], name='labels')

            # position feature
            self.input_pos1 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos1')
            self.input_pos2 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos2')

            # dropout keep probability
            self.dropout_keep_rate = tf.placeholder(tf.float32, name="dropout_keep_rate")

        with tf.name_scope('embedding_layer'):
            # embedding matrix
            self.embed_matrix_x = tf.get_variable(
                'embed_matrix_x', x_embedding.shape,
                initializer=tf.constant_initializer(x_embedding)
            )
            self.embed_size_x = int(self.embed_matrix_x.get_shape()[1])
            self.embed_matrix_pos1 = tf.get_variable('embed_matrix_pos1', [self.pos_num, self.pos_size])
            self.embed_matrix_pos2 = tf.get_variable('embed_matrix_pos2', [self.pos_num, self.pos_size])

            # embedded
            self.emb_sen = tf.nn.embedding_lookup(self.embed_matrix_x, self.input_sen)
            self.emb_pos1 = tf.nn.embedding_lookup(self.embed_matrix_pos1, self.input_pos1)
            self.emb_pos2 = tf.nn.embedding_lookup(self.embed_matrix_pos2, self.input_pos2)

            # concat embeddings
            self.emb_all = tf.concat([self.emb_sen, self.emb_pos1, self.emb_pos2], 2)

        # states and outputs
        with tf.name_scope('rnn_layer'):
            outputs_list = []
            for layer in range(2):
                with tf.name_scope('birnn_layer_{}'.format(layer)):
                    with tf.variable_scope('birnn_layer_{}'.format(layer)):
                        foward_cell = rnn_cell[self.cell_type](self.hidden_size)
                        backward_cell = rnn_cell[self.cell_type](self.hidden_size)
                        layer_output, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                            [foward_cell], [backward_cell], self.emb_all, dtype=tf.float32
                        )
                        outputs_list.append(layer_output)

            outputs = outputs_list[0] + outputs_list[1]

        with tf.name_scope('max_pooling'):
            pooled = tf.nn.max_pool(
                tf.expand_dims(outputs, -1),
                [1, self.max_sentence_len, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID'
            )
            mp_out = tf.squeeze(pooled)

        with tf.name_scope('fc_layer'):
            fc_w = tf.get_variable('fc_W', [self.hidden_size * 2, self.class_num])
            fc_b = tf.get_variable('fc_b', [self.class_num])
            self.fc_output = tf.matmul(mp_out, fc_w) + fc_b

            tf.summary.histogram('fc_W', fc_w)
            tf.summary.histogram('fc_b', fc_b)

        with tf.name_scope('softmax'):
            self.softmax_res = tf.nn.softmax(self.fc_output)

        with tf.name_scope('pred'):
            self.class_label = tf.argmax(self.softmax_res, 1)

        with tf.name_scope('model_loss'):
            self.instance_loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.fc_output, labels=self.input_labels
            )
            self.model_loss = tf.reduce_mean(self.instance_loss)

        with tf.name_scope('optimizer'):
            # optimizer
            self.optimizer = opt_method[setting.optimizer](learning_rate=setting.learning_rate).minimize(
                self.model_loss)

        # tensor board summary
        self.merge_summary = tf.summary.merge_all()

    def fit(self, session, input_data, dropout_keep_rate):
        feed_dict = {self.input_sen: input_data.x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: dropout_keep_rate
                     }
        session.run(self.optimizer, feed_dict=feed_dict)
        summary, model_loss = session.run([self.merge_summary, self.model_loss], feed_dict=feed_dict)
        return summary, model_loss

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


class BiRnn_Deep(object):
    """
    Bidirectional RNN model.
    """
    def __init__(self, x_embedding, setting):
        # model name
        self.model_name = 'BiRnn_Deep'

        # settings
        self.cell_type = setting.cells
        self.max_sentence_len = setting.sen_len
        self.hidden_sizes = setting.hidden_sizes
        self.class_num = setting.class_num
        self.pos_num = setting.pos_num
        self.pos_size = setting.pos_size
        self.learning_rate = setting.learning_rate

        with tf.name_scope('model_input'):
            # inputs
            self.input_sen = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_sen')
            self.input_labels = tf.placeholder(tf.int32, [None, self.class_num], name='labels')

            # position feature
            self.input_pos1 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos1')
            self.input_pos2 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos2')

            # dropout keep probability
            self.dropout_keep_rate = tf.placeholder(tf.float32, name="dropout_keep_rate")

        with tf.name_scope('embedding_layer'):
            # embedding matrix
            self.embed_matrix_x = tf.get_variable(
                'embed_matrix_x', x_embedding.shape,
                initializer=tf.constant_initializer(x_embedding)
            )
            self.embed_size_x = int(self.embed_matrix_x.get_shape()[1])
            self.embed_matrix_pos1 = tf.get_variable('embed_matrix_pos1', [self.pos_num, self.pos_size])
            self.embed_matrix_pos2 = tf.get_variable('embed_matrix_pos2', [self.pos_num, self.pos_size])

            # embedded
            self.emb_sen = tf.nn.embedding_lookup(self.embed_matrix_x, self.input_sen)
            self.emb_pos1 = tf.nn.embedding_lookup(self.embed_matrix_pos1, self.input_pos1)
            self.emb_pos2 = tf.nn.embedding_lookup(self.embed_matrix_pos2, self.input_pos2)

            # concat embeddings
            self.emb_all = tf.concat([self.emb_sen, self.emb_pos1, self.emb_pos2], 2)

        # states and outputs
        with tf.name_scope('rnn_layer'):
            outputs_list = []
            for layer in range(len(self.cell_type)):
                with tf.name_scope('birnn_layer_{}'.format(layer)):
                    with tf.variable_scope('birnn_layer_{}'.format(layer)):
                        foward_cell = rnn_cell[self.cell_type[layer]](self.hidden_sizes[layer])
                        backward_cell = rnn_cell[self.cell_type[layer]](self.hidden_sizes[layer])
                        layer_output, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                            [foward_cell], [backward_cell], self.emb_all, dtype=tf.float32
                        )
                        outputs_list.append(layer_output)

            self.rnn_output = tf.reduce_mean(outputs_list[-1], axis=1)
        # with tf.name_scope('max_pooling'):
        #     pooled = tf.nn.max_pool(
        #         tf.expand_dims(outputs, -1),
        #         [1, self.max_sentence_len, 1, 1],
        #         strides=[1, 1, 1, 1],
        #         padding='VALID'
        #     )
        #     mp_out = tf.squeeze(pooled)

        with tf.name_scope('fc_layer'):
            fc_w = tf.get_variable('fc_W', [self.hidden_sizes[-1] * 2, self.class_num])
            fc_b = tf.get_variable('fc_b', [self.class_num])
            self.fc_output = tf.matmul(self.rnn_output, fc_w) + fc_b

            tf.summary.histogram('fc_W', fc_w)
            tf.summary.histogram('fc_b', fc_b)

        with tf.name_scope('softmax'):
            self.softmax_res = tf.nn.softmax(self.fc_output)

        with tf.name_scope('pred'):
            self.class_label = tf.argmax(self.softmax_res, 1)

        with tf.name_scope('model_loss'):
            self.instance_loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.fc_output, labels=self.input_labels
            )
            self.model_loss = tf.reduce_mean(self.instance_loss)

        with tf.name_scope('optimizer'):
            # optimizer
            self.optimizer = opt_method[setting.optimizer](learning_rate=setting.learning_rate).minimize(
                self.model_loss)

        # tensor board summary
        self.merge_summary = tf.summary.merge_all()

    def fit(self, session, input_data, dropout_keep_rate):
        feed_dict = {self.input_sen: input_data.x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: dropout_keep_rate
                     }
        session.run(self.optimizer, feed_dict=feed_dict)
        summary, model_loss = session.run([self.merge_summary, self.model_loss], feed_dict=feed_dict)
        return summary, model_loss

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


class BiRnn_Ent(object):
    """
    Bidirectional RNN model with entity description.
    """
    def __init__(self, x_embedding, setting):
        # model name
        self.model_name = 'BiRnn_Ent'

        # settings
        self.cell_type = setting.cell
        self.max_sen_len = setting.sen_len
        self.max_ent_len = setting.ent_len
        self.hidden_size_sen = setting.hidden_size_sen
        self.hidden_size_ent = setting.hidden_size_ent
        self.class_num = setting.class_num
        self.pos_num = setting.pos_num
        self.pos_size = setting.pos_size
        self.learning_rate = setting.learning_rate

        with tf.name_scope('model_input'):
            # inputs
            self.input_sen = tf.placeholder(tf.int32, [None, self.max_sen_len], name='input_sen')
            self.input_labels = tf.placeholder(tf.int32, [None, self.class_num], name='labels')

            # position feature
            self.input_pos1 = tf.placeholder(tf.int32, [None, self.max_sen_len], name='input_pos1')
            self.input_pos2 = tf.placeholder(tf.int32, [None, self.max_sen_len], name='input_pos2')

            # entity
            self.input_e1 = tf.placeholder(tf.int32, [None, self.max_ent_len], name='input_e1')
            self.input_e2 = tf.placeholder(tf.int32, [None, self.max_ent_len], name='input_e2')

            # dropout keep probability
            self.dropout_keep_rate = tf.placeholder(tf.float32, name="dropout_keep_rate")

        with tf.name_scope('embedding_layer'):
            # embedding matrix
            self.embed_matrix_x = tf.get_variable(
                'embed_matrix_x', x_embedding.shape,
                initializer=tf.constant_initializer(x_embedding)
            )

            self.embed_matrix_ent = tf.get_variable(
                'embed_matrix_ent', x_embedding.shape,
                initializer=tf.constant_initializer(x_embedding)
            )

            self.embed_size_x = int(self.embed_matrix_x.get_shape()[1])
            self.embed_matrix_pos1 = tf.get_variable('embed_matrix_pos1', [self.pos_num, self.pos_size])
            self.embed_matrix_pos2 = tf.get_variable('embed_matrix_pos2', [self.pos_num, self.pos_size])

            # embedded
            self.emb_sen = tf.nn.embedding_lookup(self.embed_matrix_x, self.input_sen)
            self.emb_pos1 = tf.nn.embedding_lookup(self.embed_matrix_pos1, self.input_pos1)
            self.emb_pos2 = tf.nn.embedding_lookup(self.embed_matrix_pos2, self.input_pos2)
            self.emb_e1 = tf.nn.embedding_lookup(self.embed_matrix_ent, self.input_e1)
            self.emb_e2 = tf.nn.embedding_lookup(self.embed_matrix_ent, self.input_e2)

            # concat embeddings
            self.emb_all = tf.concat([self.emb_sen, self.emb_pos1, self.emb_pos2], 2)

            self.emb_e1_us = tf.unstack(self.emb_e1, num=self.max_ent_len, axis=1)
            self.emb_e2_us = tf.unstack(self.emb_e2, num=self.max_ent_len, axis=1)
            self.emb_all_us = tf.unstack(self.emb_all, num=self.max_sen_len, axis=1)

        # states and outputs
        with tf.name_scope('sentence_encoder'):
            with tf.variable_scope('sentence_encoder'):
                # cell
                foward_cell = rnn_cell[self.cell_type](self.hidden_size_sen)
                backward_cell = rnn_cell[self.cell_type](self.hidden_size_sen)
                foward_cell = tf.nn.rnn_cell.DropoutWrapper(foward_cell, output_keep_prob=self.dropout_keep_rate)
                backward_cell = tf.nn.rnn_cell.DropoutWrapper(backward_cell, output_keep_prob=self.dropout_keep_rate)

                # rnn
                sen_output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                    foward_cell, backward_cell, self.emb_all_us, dtype=tf.float32
                )

                if setting.hidden_select == 'last':
                    self.sent_output = sen_output[-1]
                elif setting.hidden_select == 'avg':
                    self.sent_output = tf.reduce_mean(
                        tf.reshape(tf.concat(sen_output, 1), [-1, self.max_sen_len, self.hidden_size_sen * 2]),
                        axis=1
                    )

        with tf.name_scope('entity_encoder'):
            with tf.variable_scope('entity_encoder'):
                # cell
                foward_cell = rnn_cell[self.cell_type](self.hidden_size_sen)
                backward_cell = rnn_cell[self.cell_type](self.hidden_size_sen)
                # rnn
                ent1_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                    foward_cell, backward_cell, self.emb_e1_us, dtype=tf.float32
                )
                ent2_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                    foward_cell, backward_cell, self.emb_e2_us, dtype=tf.float32
                )
                # entity representation
                ent1_output = tf.reduce_mean(
                    tf.reshape(tf.concat(ent1_outputs, 1), [-1, self.max_ent_len, self.hidden_size_ent * 2]),
                    axis=1
                )
                ent2_output = tf.reduce_mean(
                    tf.reshape(tf.concat(ent2_outputs, 1), [-1, self.max_ent_len, self.hidden_size_ent * 2]),
                    axis=1
                )
                self.ent_out = tf.concat([ent1_output, ent2_output], axis=1)

        with tf.name_scope('joint_layer'):
            self.rep = tf.concat([self.sent_output, self.ent_out], axis=1)

        with tf.name_scope('fc_layer'):
            self.fc_w = tf.get_variable('fc_W', [self.hidden_size_sen * 2 + self.hidden_size_ent * 4, self.class_num])
            self.fc_b = tf.get_variable('fc_b', [self.class_num])
            self.fc_output = tf.matmul(self.rep, self.fc_w) + self.fc_b

            tf.summary.histogram('fc_W', self.fc_w)
            tf.summary.histogram('fc_b', self.fc_b)

        with tf.name_scope('softmax'):
            self.softmax_res = tf.nn.softmax(self.fc_output)

        with tf.name_scope('model_predict'):
            self.class_label = tf.argmax(self.softmax_res, 1)

        with tf.name_scope('model_loss'):
            # choose the min loss instance index
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc_output, labels=self.input_labels)
            # model loss
            self.model_loss = tf.reduce_mean(self.loss)

            tf.summary.scalar('model_loss', self.model_loss)

        with tf.name_scope('optimizer'):
            # optimizer
            self.optimizer = opt_method[setting.optimizer](learning_rate=setting.learning_rate).minimize(
                self.model_loss)

        # tensor board summary
        self.merge_summary = tf.summary.merge_all()

    def fit(self, session, input_data, dropout_keep_rate):
        feed_dict = {self.input_sen: input_data.x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_e1: input_data.e1,
                     self.input_e2: input_data.e2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: dropout_keep_rate
                     }
        session.run(self.optimizer, feed_dict=feed_dict)
        summary, model_loss = session.run([self.merge_summary, self.model_loss], feed_dict=feed_dict)
        return summary, model_loss

    def evaluate(self, session, input_data):
        feed_dict = {self.input_sen: input_data.x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_e1: input_data.e1,
                     self.input_e2: input_data.e2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: 1}
        model_loss, label_pred, label_prob = session.run(
            [self.model_loss, self.class_label, self.softmax_res], feed_dict=feed_dict
        )
        return model_loss, label_pred, label_prob


class BiRnn_Att_Ent(object):
    """
    Bidirectional RNN model with attention and entity spell.
    """

    def __init__(self, x_embedding, setting):
        # model name
        self.model_name = 'BiRnn_Att_Ent'

        # settings
        self.cell_type = setting.cell
        self.max_sentence_len = setting.sen_len
        self.max_ent_len = setting.ent_len
        self.hidden_size_sen = setting.hidden_size_sen
        self.hidden_size_ent = setting.hidden_size_ent
        self.class_num = setting.class_num
        self.pos_num = setting.pos_num
        self.pos_size = setting.pos_size
        self.learning_rate = setting.learning_rate

        with tf.name_scope('model_input'):
            # inputs
            self.input_sen = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_sen')
            self.input_labels = tf.placeholder(tf.int32, [None, self.class_num], name='labels')

            # position feature
            self.input_pos1 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos1')
            self.input_pos2 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos2')

            # entity
            self.input_e1 = tf.placeholder(tf.int32, [None, self.max_ent_len], name='input_e1')
            self.input_e2 = tf.placeholder(tf.int32, [None, self.max_ent_len], name='input_e2')

            # dropout keep probability
            self.dropout_keep_rate = tf.placeholder(tf.float32, name="dropout_keep_rate")

        with tf.name_scope('embedding_layer'):
            # embedding matrix
            self.embed_matrix_x = tf.get_variable(
                'embed_matrix_x', x_embedding.shape,
                initializer=tf.constant_initializer(x_embedding)
            )
            self.embed_size_x = int(self.embed_matrix_x.get_shape()[1])
            self.embed_matrix_pos1 = tf.get_variable('embed_matrix_pos1', [self.pos_num, self.pos_size])
            self.embed_matrix_pos2 = tf.get_variable('embed_matrix_pos2', [self.pos_num, self.pos_size])

            # embedded
            self.emb_sen = tf.nn.embedding_lookup(self.embed_matrix_x, self.input_sen)
            self.emb_pos1 = tf.nn.embedding_lookup(self.embed_matrix_pos1, self.input_pos1)
            self.emb_pos2 = tf.nn.embedding_lookup(self.embed_matrix_pos2, self.input_pos2)
            self.emb_e1 = tf.nn.embedding_lookup(self.embed_matrix_x, self.input_e1)
            self.emb_e2 = tf.nn.embedding_lookup(self.embed_matrix_x, self.input_e2)

            # concat embeddings
            self.emb_all = tf.concat([self.emb_sen, self.emb_pos1, self.emb_pos2], 2)

            self.emb_e1_us = tf.unstack(self.emb_e1, num=self.max_ent_len, axis=1)
            self.emb_e2_us = tf.unstack(self.emb_e2, num=self.max_ent_len, axis=1)
            self.emb_all_us = tf.unstack(self.emb_all, num=self.max_sentence_len, axis=1)

        # states and outputs
        with tf.name_scope('sentence_encoder'):
            with tf.variable_scope('sentence_encoder'):
                # cell
                foward_cell = rnn_cell[self.cell_type](self.hidden_size_sen)
                backward_cell = rnn_cell[self.cell_type](self.hidden_size_sen)
                foward_cell = tf.nn.rnn_cell.DropoutWrapper(foward_cell, output_keep_prob=self.dropout_keep_rate)
                backward_cell = tf.nn.rnn_cell.DropoutWrapper(backward_cell, output_keep_prob=self.dropout_keep_rate)

                # rnn
                rnn_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                    foward_cell, backward_cell, self.emb_all_us, dtype=tf.float32
                )

                self.sent_outputs = tf.reshape(tf.concat(rnn_outputs, axis=1),
                                               [-1, self.max_sentence_len, self.hidden_size_sen * 2])

        with tf.name_scope('entity_encoder'):
            with tf.variable_scope('entity_encoder'):
                # cell
                foward_cell = rnn_cell[self.cell_type](self.hidden_size_sen)
                backward_cell = rnn_cell[self.cell_type](self.hidden_size_sen)

                # rnn
                ent1_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                    foward_cell, backward_cell, self.emb_e1_us, dtype=tf.float32
                )
                e1_outputs_forward = [i[:, :self.hidden_size_ent] for i in ent1_outputs]
                e1_outputs_backward = [i[:, self.hidden_size_ent:] for i in ent1_outputs]
                e1_outputs_forward = tf.reshape(tf.concat(axis=1, values=e1_outputs_forward),
                                                [-1, self.max_ent_len, self.hidden_size_ent])
                e1_outputs_backward = tf.reshape(tf.concat(axis=1, values=e1_outputs_backward),
                                                 [-1, self.max_ent_len, self.hidden_size_ent])
                e1_outputs = tf.add(e1_outputs_forward, e1_outputs_backward)
                e1_output = tf.reduce_mean(e1_outputs, 1)

                ent2_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                    foward_cell, backward_cell, self.emb_e2_us, dtype=tf.float32
                )
                e2_outputs_forward = [i[:, :self.hidden_size_ent] for i in ent2_outputs]
                e2_outputs_backward = [i[:, self.hidden_size_ent:] for i in ent2_outputs]
                e2_output_forward = tf.reshape(tf.concat(axis=1, values=e2_outputs_forward),
                                               [-1, self.max_ent_len, self.hidden_size_ent])
                e2_output_backward = tf.reshape(tf.concat(axis=1, values=e2_outputs_backward),
                                                [-1, self.max_ent_len, self.hidden_size_ent])
                e2_outputs = tf.add(e2_output_forward, e2_output_backward)
                e2_output = tf.reduce_mean(e2_outputs, 1)

                self.ent_out = tf.concat([e1_output, e2_output], axis=1)
                self.ent_att = tf.expand_dims(tf.subtract(e1_output, e2_output), -1)

        with tf.name_scope('attention_layer'):
            self.attention_w = tf.get_variable('attention_omega', [self.sent_outputs.shape[-1], self.ent_att.shape[-2]])
            self.attention_A = tf.reshape(
                tf.nn.softmax(
                    tf.squeeze(
                        tf.matmul(
                            tf.reshape(
                                tf.matmul(
                                    tf.reshape(tf.tanh(self.sent_outputs), [-1, int(self.sent_outputs.shape[-1])]),
                                    self.attention_w
                                ),
                                [-1, self.max_sentence_len, self.hidden_size_ent]
                            ),
                            self.ent_att
                        ),
                    )
                ),
                [-1, 1, self.max_sentence_len]
            )
            self.att_output = tf.reshape(tf.matmul(self.attention_A, self.sent_outputs),
                                         [-1, int(self.sent_outputs.shape[-1])])

        with tf.name_scope('joint_layer'):
            self.rep = tf.concat([self.att_output, self.ent_out], axis=1)

        with tf.name_scope('fc_layer'):
            self.fc_w = tf.get_variable('fc_W',
                                        [int(self.sent_outputs.shape[-1]) + self.hidden_size_ent * 2, self.class_num])
            self.fc_b = tf.get_variable('fc_b', [self.class_num])
            self.fc_output = tf.matmul(self.rep, self.fc_w) + self.fc_b

            tf.summary.histogram('fc_W', self.fc_w)
            tf.summary.histogram('fc_b', self.fc_b)

        with tf.name_scope('softmax'):
            self.softmax_res = tf.nn.softmax(self.fc_output)

        with tf.name_scope('model_predict'):
            self.class_label = tf.argmax(self.softmax_res, 1)

        with tf.name_scope('model_loss'):
            # choose the min loss instance index
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc_output, labels=self.input_labels)
            # model loss
            self.model_loss = tf.reduce_mean(self.loss)

            tf.summary.scalar('model_loss', self.model_loss)

        with tf.name_scope('optimizer'):
            # optimizer
            self.optimizer = opt_method[setting.optimizer](learning_rate=setting.learning_rate).minimize(
                self.model_loss)

        # tensor board summary
        self.merge_summary = tf.summary.merge_all()

    def fit(self, session, input_data, dropout_keep_rate):
        feed_dict = {self.input_sen: input_data.x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_e1: input_data.e1,
                     self.input_e2: input_data.e2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: dropout_keep_rate
                     }
        session.run(self.optimizer, feed_dict=feed_dict)
        summary, model_loss = session.run([self.merge_summary, self.model_loss], feed_dict=feed_dict)
        return summary, model_loss

    def evaluate(self, session, input_data):
        feed_dict = {self.input_sen: input_data.x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_e1: input_data.e1,
                     self.input_e2: input_data.e2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: 1}
        model_loss, label_pred, label_prob = session.run(
            [self.model_loss, self.class_label, self.softmax_res], feed_dict=feed_dict
        )
        return model_loss, label_pred, label_prob


class BiRnn_Cnn_Ent(object):
    """
    A model use birnn and cnn.
    """
    def __init__(self, x_embedding, setting):
        # model name
        self.model_name = 'BiRnn_Cnn_Ent'

        # settings
        self.cell_type = setting.cell
        self.max_sentence_len = setting.sen_len
        self.max_ent_len = setting.ent_len
        self.hidden_size_sen = setting.hidden_size_sen
        self.hidden_size_ent = setting.hidden_size_ent
        self.filter_sizes = setting.filter_sizes
        self.filter_num = setting.filter_num
        self.class_num = setting.class_num
        self.pos_num = setting.pos_num
        self.pos_size = setting.pos_size
        self.learning_rate = setting.learning_rate

        with tf.name_scope('model_input'):
            # inputs
            self.input_sen = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_sen')
            self.input_labels = tf.placeholder(tf.int32, [None, self.class_num], name='labels')

            # position feature
            self.input_pos1 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos1')
            self.input_pos2 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='input_pos2')

            # entity
            self.input_e1 = tf.placeholder(tf.int32, [None, self.max_ent_len], name='input_e1')
            self.input_e2 = tf.placeholder(tf.int32, [None, self.max_ent_len], name='input_e2')

            # dropout keep probability
            self.dropout_keep_rate = tf.placeholder(tf.float32, name="dropout_keep_rate")

        with tf.name_scope('embedding_layer'):
            # embedding matrix
            self.embed_matrix_x = tf.get_variable(
                'embed_matrix_x', x_embedding.shape,
                initializer=tf.constant_initializer(x_embedding)
            )
            self.embed_size_x = int(self.embed_matrix_x.get_shape()[1])
            self.embed_matrix_pos1 = tf.get_variable('embed_matrix_pos1', [self.pos_num, self.pos_size])
            self.embed_matrix_pos2 = tf.get_variable('embed_matrix_pos2', [self.pos_num, self.pos_size])

            # embedded
            self.emb_sen = tf.nn.embedding_lookup(self.embed_matrix_x, self.input_sen)
            self.emb_pos1 = tf.nn.embedding_lookup(self.embed_matrix_pos1, self.input_pos1)
            self.emb_pos2 = tf.nn.embedding_lookup(self.embed_matrix_pos2, self.input_pos2)
            self.emb_e1 = tf.nn.embedding_lookup(self.embed_matrix_x, self.input_e1)
            self.emb_e2 = tf.nn.embedding_lookup(self.embed_matrix_x, self.input_e2)

            self.emb_sen_us = tf.unstack(self.emb_sen, num=self.max_sentence_len, axis=1)
            self.emb_e1_us = tf.unstack(self.emb_e1, num=self.max_ent_len, axis=1)
            self.emb_e2_us = tf.unstack(self.emb_e2, num=self.max_ent_len, axis=1)

        # states and outputs
        with tf.name_scope('sentence_encoder'):
            with tf.name_scope('rnn_layer'):
                with tf.variable_scope('sentence_encoder'):
                    # cell
                    foward_cell = rnn_cell[self.cell_type](self.hidden_size_sen)
                    backward_cell = rnn_cell[self.cell_type](self.hidden_size_sen)
                    foward_cell = tf.nn.rnn_cell.DropoutWrapper(foward_cell, output_keep_prob=self.dropout_keep_rate)
                    backward_cell = tf.nn.rnn_cell.DropoutWrapper(backward_cell, output_keep_prob=self.dropout_keep_rate)

                    # rnn
                    sen_rnn_output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                        foward_cell, backward_cell, self.emb_sen_us, dtype=tf.float32
                    )
                    self.sen_rnn_output = tf.reshape(tf.concat(sen_rnn_output, axis=1),
                                                     [-1, self.max_sentence_len, self.hidden_size_sen * 2])

            with tf.name_scope('add_pos'):
                # concat embeddings
                self.cnn_input = tf.expand_dims(tf.concat([self.sen_rnn_output, self.emb_pos1, self.emb_pos2], 2), -1)

            with tf.name_scope('conv_maxpooling'):
                # convolution and max pooling
                pooled_outputs = []
                for i, filter_size in enumerate(self.filter_sizes):
                    with tf.name_scope('conv-maxpool-{}'.format(filter_size)):
                        # convolution layer
                        filter_shape = [filter_size, self.hidden_size_sen * 2 + self.pos_size * 2, 1, self.filter_num]

                        w = tf.get_variable('W_{}'.format(filter_size),
                                            filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
                        b = tf.get_variable('b_{}'.format(filter_size),
                                            [self.filter_num], initializer=tf.constant_initializer(0.1))

                        tf.summary.histogram('W_{}'.format(filter_size), w)
                        tf.summary.histogram('b_{}'.format(filter_size), b)

                        conv = tf.nn.conv2d(
                            self.cnn_input, w, strides=[1, 1, 1, 1], padding='VALID', name='conv'
                        )

                        # Apply none linearity
                        h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                        # Max pooling over the outputs
                        pooled = tf.nn.max_pool(
                            h,
                            ksize=[1, self.max_sentence_len - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID', name='conv'
                        )
                        pooled_outputs.append(pooled)

                # Combine all the pooled features
                num_filters_total = self.filter_num * len(self.filter_sizes)
                h_pool = tf.concat(pooled_outputs, 3)
                h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

                # Add dropout
                self.sen_output = tf.nn.dropout(h_pool_flat, self.dropout_keep_rate)

        with tf.name_scope('entity_encoder'):
            with tf.variable_scope('entity_encoder'):
                # cell
                foward_cell = rnn_cell[self.cell_type](self.hidden_size_ent)
                backward_cell = rnn_cell[self.cell_type](self.hidden_size_ent)
                # rnnv
                ent1_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                    foward_cell, backward_cell, self.emb_e1_us, dtype=tf.float32
                )
                ent2_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                    foward_cell, backward_cell, self.emb_e2_us, dtype=tf.float32
                )
                # entity representation
                ent1_output = tf.reduce_mean(
                    tf.reshape(tf.concat(ent1_outputs, 1), [-1, self.max_ent_len, self.hidden_size_ent * 2]),
                    axis=1
                )
                ent2_output = tf.reduce_mean(
                    tf.reshape(tf.concat(ent2_outputs, 1), [-1, self.max_ent_len, self.hidden_size_ent * 2]),
                    axis=1
                )
                self.ent_out = tf.concat([ent1_output, ent2_output], axis=1)

        with tf.name_scope('joint_layer'):
            self.outputs = tf.concat([self.sen_output, self.ent_out], axis=1)

        # full connection layer
        with tf.name_scope('fc_layer'):
            # full connection layer before softmax
            self.fc_w = tf.get_variable(
                'fc_W',
                [self.filter_num * len(self.filter_sizes) + self.hidden_size_ent * 4, self.class_num]
            )
            self.fc_b = tf.get_variable('fc_b', [self.class_num])

            tf.summary.histogram('fc_W', self.fc_w)
            tf.summary.histogram('fc_b', self.fc_b)

            self.fc_out = tf.matmul(self.outputs, self.fc_w) + self.fc_b

        # softmax
        with tf.name_scope('softmax'):
            self.softmax_res = tf.nn.softmax(self.fc_out)

        with tf.name_scope("accuracy"):
            # get max softmax predict result of each relation
            self.maxres_by_rel = tf.reduce_max(self.softmax_res, 0)

            # class label
            self.class_label = tf.argmax(self.softmax_res, 1)

            # accuracy
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.class_label, tf.argmax(self.input_labels, 1)), "float"), name="accuracy"
            )
            tf.summary.scalar('accuracy', self.accuracy)

        with tf.name_scope('model_predict'):
            # get max softmax predict result of each relation
            self.maxres_by_rel = tf.reduce_max(self.softmax_res, 0)

            # class label
            self.class_label = tf.argmax(self.softmax_res, 1)

        with tf.name_scope('model_loss'):
            # choose the min loss instance index
            self.instance_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc_out, labels=self.input_labels)

            # model loss
            self.model_loss = tf.reduce_mean(self.instance_loss)
            tf.summary.scalar('model_loss', self.model_loss)

        with tf.name_scope('optimizer'):
            # optimizer
            self.optimizer = opt_method[setting.optimizer](learning_rate=setting.learning_rate).minimize(
                self.model_loss)

        # tensor board summary
        self.merge_summary = tf.summary.merge_all()

    def fit(self, session, input_data, dropout_keep_rate):
        feed_dict = {self.input_sen: input_data.x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_e1: input_data.e1,
                     self.input_e2: input_data.e2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: dropout_keep_rate
                     }
        session.run(self.optimizer, feed_dict=feed_dict)
        summary, model_loss = session.run([self.merge_summary, self.model_loss], feed_dict=feed_dict)
        return summary, model_loss

    def evaluate(self, session, input_data):
        feed_dict = {self.input_sen: input_data.x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_e1: input_data.e1,
                     self.input_e2: input_data.e2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: 1}
        model_loss, label_pred, label_prob = session.run(
            [self.model_loss, self.class_label, self.softmax_res], feed_dict=feed_dict
        )
        return model_loss, label_pred, label_prob


class BiRnn_Res_Ent(object):
    """
    Bidirectional RNN model.
    """
    def __init__(self, x_embedding, setting):
        # model name
        self.model_name = 'BiRnn_Res_Ent'

        # settings
        self.cell_type = setting.cell
        self.max_sen_len = setting.sen_len
        self.max_ent_len = setting.ent_len
        self.hidden_size_sen = setting.hidden_size_sen
        self.hidden_size_ent = setting.hidden_size_ent
        self.class_num = setting.class_num
        self.pos_num = setting.pos_num
        self.pos_size = setting.pos_size
        self.learning_rate = setting.learning_rate

        with tf.name_scope('model_input'):
            # inputs
            self.input_sen = tf.placeholder(tf.int32, [None, self.max_sen_len], name='input_sen')
            self.input_labels = tf.placeholder(tf.int32, [None, self.class_num], name='labels')

            # position feature
            self.input_pos1 = tf.placeholder(tf.int32, [None, self.max_sen_len], name='input_pos1')
            self.input_pos2 = tf.placeholder(tf.int32, [None, self.max_sen_len], name='input_pos2')

            # entity
            self.input_e1 = tf.placeholder(tf.int32, [None, self.max_ent_len], name='input_e1')
            self.input_e2 = tf.placeholder(tf.int32, [None, self.max_ent_len], name='input_e2')

            # dropout keep probability
            self.dropout_keep_rate = tf.placeholder(tf.float32, name="dropout_keep_rate")

        with tf.name_scope('embedding_layer'):
            # embedding matrix
            self.embed_matrix_x = tf.get_variable(
                'embed_matrix_x', x_embedding.shape,
                initializer=tf.constant_initializer(x_embedding)
            )
            self.embed_size_x = int(self.embed_matrix_x.get_shape()[1])
            self.embed_matrix_pos1 = tf.get_variable('embed_matrix_pos1', [self.pos_num, self.pos_size])
            self.embed_matrix_pos2 = tf.get_variable('embed_matrix_pos2', [self.pos_num, self.pos_size])

            # embedded
            self.emb_sen = tf.nn.embedding_lookup(self.embed_matrix_x, self.input_sen)
            self.emb_pos1 = tf.nn.embedding_lookup(self.embed_matrix_pos1, self.input_pos1)
            self.emb_pos2 = tf.nn.embedding_lookup(self.embed_matrix_pos2, self.input_pos2)
            self.emb_e1 = tf.nn.embedding_lookup(self.embed_matrix_x, self.input_e1)
            self.emb_e2 = tf.nn.embedding_lookup(self.embed_matrix_x, self.input_e2)

            # concat embeddings
            self.emb_all = tf.concat([self.emb_sen, self.emb_pos1, self.emb_pos2], 2)

        with tf.name_scope('sentence_encoder'):
            # states and outputs
            with tf.name_scope('rnn_layer'):
                outputs_list = []
                for layer in range(2):
                    with tf.name_scope('birnn_layer_{}'.format(layer)):
                        with tf.variable_scope('birnn_layer_{}'.format(layer)):
                            foward_cell = rnn_cell[self.cell_type](self.hidden_size_sen)
                            backward_cell = rnn_cell[self.cell_type](self.hidden_size_sen)
                            layer_output, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                                [foward_cell], [backward_cell], self.emb_all, dtype=tf.float32
                            )
                            outputs_list.append(layer_output)

                outputs = outputs_list[0] + outputs_list[1]

            with tf.name_scope('max_pooling'):
                pooled = tf.nn.max_pool(
                    tf.expand_dims(outputs, -1),
                    [1, self.max_sen_len, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID'
                )
                mp_out = tf.squeeze(pooled)
                self.sen_output = mp_out

        with tf.name_scope('entity_encoder'):
            with tf.variable_scope('entity_encoder'):
                # cell
                foward_cell = rnn_cell[self.cell_type](self.hidden_size_sen)
                backward_cell = rnn_cell[self.cell_type](self.hidden_size_sen)
                # rnn
                ent1_outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    [foward_cell], [backward_cell], self.emb_e1, dtype=tf.float32
                )
                ent2_outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    [foward_cell], [backward_cell], self.emb_e2, dtype=tf.float32
                )
                # entity representation
                ent1_output = tf.reduce_mean(ent1_outputs, axis=1)
                ent2_output = tf.reduce_mean(ent2_outputs, axis=1)
                self.ent_out = tf.concat([ent1_output, ent2_output], axis=1)

        with tf.name_scope('joint_layer'):
            self.rep = tf.concat([self.sen_output, self.ent_out], axis=1)

        with tf.name_scope('fc_layer'):
            fc_w = tf.get_variable('fc_W', [self.hidden_size_sen * 2 + self.hidden_size_ent * 4, self.class_num])
            fc_b = tf.get_variable('fc_b', [self.class_num])
            self.fc_output = tf.matmul(self.rep, fc_w) + fc_b

            tf.summary.histogram('fc_W', fc_w)
            tf.summary.histogram('fc_b', fc_b)

        with tf.name_scope('softmax'):
            self.softmax_res = tf.nn.softmax(self.fc_output)

        with tf.name_scope('pred'):
            self.class_label = tf.argmax(self.softmax_res, 1)

        with tf.name_scope('model_loss'):
            self.instance_loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.fc_output, labels=self.input_labels
            )
            self.model_loss = tf.reduce_mean(self.instance_loss)

        with tf.name_scope('optimizer'):
            # optimizer
            self.optimizer = opt_method[setting.optimizer](learning_rate=setting.learning_rate).minimize(
                self.model_loss)

        # tensor board summary
        self.merge_summary = tf.summary.merge_all()

    def fit(self, session, input_data, dropout_keep_rate):
        feed_dict = {self.input_sen: input_data.x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_e1: input_data.e1,
                     self.input_e2: input_data.e2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: dropout_keep_rate
                     }
        session.run(self.optimizer, feed_dict=feed_dict)
        summary, model_loss = session.run([self.merge_summary, self.model_loss], feed_dict=feed_dict)
        return summary, model_loss

    def evaluate(self, session, input_data):
        feed_dict = {self.input_sen: input_data.x,
                     self.input_pos1: input_data.pos1,
                     self.input_pos2: input_data.pos2,
                     self.input_e1: input_data.e1,
                     self.input_e2: input_data.e2,
                     self.input_labels: input_data.y,
                     self.dropout_keep_rate: 1}
        model_loss, label_pred, label_prob = session.run(
            [self.model_loss, self.class_label, self.softmax_res], feed_dict=feed_dict
        )
        return model_loss, label_pred, label_prob


class BiRnn_Mi(object):
    """
    Bidirectional RNN model with multi-instance learning.
    """
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

        with tf.name_scope('input_layer'):
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

        with tf.name_scope('embedding_layer'):
            # embedding matrix
            self.embed_matrix_x = tf.get_variable(
                'embed_matrix_x', x_embedding.shape,
                initializer=tf.constant_initializer(x_embedding)
            )
            self.embed_size_x = int(self.embed_matrix_x.get_shape()[1])
            self.embed_matrix_pos1 = tf.get_variable('embed_matrix_pos1', [self.pos_num, self.pos_size])
            self.embed_matrix_pos2 = tf.get_variable('embed_matrix_pos2', [self.pos_num, self.pos_size])

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
            if setting.hidden_select == 'last':
                self.sen_emb = self.outputs[-1]
            elif setting.hidden_select == 'avg':
                self.sen_emb = tf.reduce_mean(
                    tf.reshape(tf.concat(self.outputs, 1), [-1, self.max_sentence_len, self.hidden_size * 2]), axis=1
                )

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

            self.sen_a = tf.get_variable('attention_A', [self.hidden_size * 2])
            self.sen_r = tf.get_variable('query_r', [self.hidden_size * 2, 1])
            relation_embedding = tf.get_variable('relation_embedding', [self.class_num, self.hidden_size * 2])
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

                sen_s.append(tf.reshape(tf.matmul(sen_alpha[i], sen_repre[i]), [self.hidden_size * 2, 1]))
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

        with tf.name_scope('optimizer'):
            # optimizer
            self.optimizer = opt_method[setting.optimizer](learning_rate=setting.learning_rate).minimize(
                self.total_loss)

        # tensor board summary
        tf.summary.histogram('sen_a', self.sen_a)
        tf.summary.histogram('sen_r', self.sen_r)
        tf.summary.scalar('loss', self.total_loss)
        self.merge_summary = tf.summary.merge_all()

    def fit(self, session, input_data, dropout_keep_rate):
        total_shape = [0]
        total_num = 0
        total_x = []
        total_pos1 = []
        total_pos2 = []
        for bag_idx in range(len(input_data.x)):
            total_num += len(input_data.x[bag_idx])
            total_shape.append(total_num)
            for sent in input_data.x[bag_idx]:
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
        summary, model_accuracy, model_loss = session.run([self.merge_summary, self.accuracy, self.total_loss],
                                                          feed_dict=feed_dict)
        return summary, model_loss

    def evaluate(self, session, input_data):
        total_shape = [0]
        total_num = 0
        total_x = []
        total_pos1 = []
        total_pos2 = []
        for bag_idx in range(len(input_data.x)):
            total_num += len(input_data.x[bag_idx])
            total_shape.append(total_num)
            for sent in input_data.x[bag_idx]:
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
            self.dropout_keep_rate: 1
        }
        model_loss, label_pred, label_prob = session.run(
            [self.total_loss, self.predictions, self.prob], feed_dict=feed_dict
        )
        return model_loss, label_pred, label_prob
