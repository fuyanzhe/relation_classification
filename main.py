# -*- encoding: utf-8 -*-
# Created by han on 17-7-8

from data_loader import DataLoader
from model_settings import *
from models import *
from evaluate import *


def main():
    model_name = 'birnn_att'
    train_epochs_num = 100

    if model_name == 'cnn':
        # cnn test
        data_loader = DataLoader('./data', multi_ins=False)
        cnn_setting = CNNSetting()
        cnn_model = CNN(data_loader.wordembedding, cnn_setting)
        print cnn_model.model_name

        with tf.Session() as session:
            tf.global_variables_initializer().run()
            test_data = data_loader.get_test_data()
            for epoch_num in range(train_epochs_num):
                iter_num = 0
                batches = data_loader.get_train_batches(batch_size=1024)
                for batch in batches:
                    iter_num += 1
                    loss = cnn_model.fit(session, batch, dropout_keep_rate=0.5)
                    _, c_label = cnn_model.evaluate(session, batch)
                    if iter_num % 100 == 0:
                        p, r, f1 = get_p_r_f1(c_label, batch.y)
                        print 'epoch: {:>3}, batch: {:>4}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
                            epoch_num, iter_num, loss, p * 100, r * 100, f1 * 100
                        )
                test_loss, test_pred = cnn_model.evaluate(session, test_data)
                p, r, f1 = get_p_r_f1(test_pred, test_data.y)
                print 'epoch: {:>3}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
                    epoch_num, test_loss, p * 100, r * 100, f1 * 100
                )
    elif model_name == 'rnn':
        # rnn test
        data_loader = DataLoader('./data', multi_ins=False)
        lstm_setting = RNNSetting()
        lstm_model = RNN(data_loader.wordembedding, lstm_setting)

        with tf.Session() as session:
            tf.global_variables_initializer().run()
            test_data = data_loader.get_test_data()
            for epoch_num in range(100):
                iter_num = 0
                batches = data_loader.get_train_batches(batch_size=1024)
                for batch in batches:
                    iter_num += 1
                    loss = lstm_model.fit(session, batch, dropout_keep_rate=0.5)
                    _, c_label = lstm_model.evaluate(session, batch)
                    if iter_num % 100 == 0:
                        p, r, f1 = get_p_r_f1(c_label, batch.y)
                        print 'epoch: {:>3}, batch: {:>4}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
                            epoch_num, iter_num, loss, p * 100, r * 100, f1 * 100
                        )
                test_loss, test_pred = lstm_model.evaluate(session, test_data)
                p, r, f1 = get_p_r_f1(test_pred, test_data.y)
                print 'epoch: {:>3}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
                    epoch_num, test_loss, p * 100, r * 100, f1 * 100
                )
    elif model_name == 'birnn':
        # bigru test
        data_loader = DataLoader('./data', multi_ins=False)
        bigru_setting = RNNSetting()
        bigru_model = BiRNN(data_loader.wordembedding, bigru_setting)

        with tf.Session() as session:
            tf.global_variables_initializer().run()
            test_data = data_loader.get_test_data()
            for epoch_num in range(100):
                iter_num = 0
                batches = data_loader.get_train_batches(batch_size=1024)
                for batch in batches:
                    iter_num += 1
                    loss = bigru_model.fit(session, batch, dropout_keep_rate=0.5)
                    _, c_label = bigru_model.evaluate(session, batch)
                    if iter_num % 100 == 0:
                        p, r, f1 = get_p_r_f1(c_label, batch.y)
                        print 'epoch: {:>3}, batch: {:>4}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
                            epoch_num, iter_num, loss, p * 100, r * 100, f1 * 100
                        )
                test_loss, test_pred = bigru_model.evaluate(session, test_data)
                p, r, f1 = get_p_r_f1(test_pred, test_data.y)
                print 'epoch: {:>3}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
                    epoch_num, test_loss, p * 100, r * 100, f1 * 100
                )
    elif model_name == 'birnn_att':
        # bigru_att test
        data_loader = DataLoader('./data', multi_ins=False)
        rnn_att_setting = RNNSetting()
        rnn_att_model = BiRNN_ATT(data_loader.wordembedding, rnn_att_setting)

        # with tf.Session() as session:
        #     tf.global_variables_initializer().run()
        #     for i in range(100):
        #         batches = data_loader.get_train_batches(batch_size=128)
        #         for batch in batches:
        #             feed_dict = {rnn_att_model.input_words: batch.word,
        #                          rnn_att_model.input_pos1: batch.pos1,
        #                          rnn_att_model.input_pos2: batch.pos2,
        #                          rnn_att_model.input_labels: batch.y,
        #                          rnn_att_model.dropout_keep_rate: 0.5
        #                          }
        #             A, w, out_h, out_final = session.run(
        #                 [rnn_att_model.attention_A, rnn_att_model.attention_w, rnn_att_model.output_h, rnn_att_model.output_final],
        #                 feed_dict=feed_dict
        #             )
        #             # rnn_att_model.fit(session, batch, 0.5)
        #             pass
        with tf.Session() as session:
            tf.global_variables_initializer().run()
            test_data = data_loader.get_test_data()
            for epoch_num in range(100):
                iter_num = 0
                batches = data_loader.get_train_batches(batch_size=1024)
                for batch in batches:
                    iter_num += 1
                    loss = rnn_att_model.fit(session, batch, dropout_keep_rate=0.5)
                    _, c_label = rnn_att_model.evaluate(session, batch)
                    if iter_num % 100 == 0:
                        p, r, f1 = get_p_r_f1(c_label, batch.y)
                        print 'epoch: {:>3}, batch: {:>4}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
                            epoch_num, iter_num, loss, p * 100, r * 100, f1 * 100
                        )
                test_loss, test_pred = rnn_att_model.evaluate(session, test_data)
                p, r, f1 = get_p_r_f1(test_pred, test_data.y)
                print 'epoch: {:>3}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
                    epoch_num, test_loss, p * 100, r * 100, f1 * 100
                )
    elif model_name == 'rnn_mi':
        # bigru _mi test
        data_loader = DataLoader('./data', multi_ins=True)
        rnn_mi_setting = RNNSetting()
        rnn_mi_model = RNN_MI(data_loader.wordembedding, rnn_mi_setting)

        with tf.Session() as session:
            tf.global_variables_initializer().run()
            for i in range(100):
                batches = data_loader.get_train_batches(batch_size=128)
                for batch in batches:
                    # feed_dict = {bigru_model.input_words: batch.word,
                    #              bigru_model.input_pos1: batch.pos1,
                    #              bigru_model.input_pos2: batch.pos2,
                    #              bigru_model.input_labels: batch.y,
                    #              bigru_model.dropout_keep_rate: 0.5
                    #              }
                    # op, opf, opb = session.run([bigru_model.outputs], feed_dict=feed_dict)
                    rnn_mi_model.fit(session, batch, 0.5)
                    pass

if __name__ == '__main__':
    main()
