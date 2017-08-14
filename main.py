# -*- encoding: utf-8 -*-
# Created by han on 17-7-8

from models import *
from data_loader import DataLoader
from evaluate import *


def main():
    data_loader = DataLoader('./data', multi_ins=False)

    # # cnn test
    # cnn_setting = CNNSetting()
    # cnn_model = CNN(data_loader.wordembedding, cnn_setting)
    # print cnn_model.model_name
    #
    # with tf.Session() as session:
    #     tf.global_variables_initializer().run()
    #     test_data = data_loader.get_test_data()
    #     for i in range(100):
    #         j = 0
    #         batches = data_loader.get_train_batches(batch_size=1024)
    #         for batch in batches:
    #             j += 1
    #             loss = cnn_model.fit(session, batch, dropout_keep_rate=0.5)
    #             _, c_label = cnn_model.predict(session, batch)
    #             if j % 100 == 0:
    #                 p, r, f1 = get_p_r_f1(c_label, batch.y)
    #                 print 'epoch: {:>3}, batch: {:>4}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
    #                     i, j, loss, p * 100, r * 100, f1 * 100
    #                 )
    #         test_loss, test_pred = cnn_model.predict(session, test_data)
    #         p, r, f1 = get_p_r_f1(test_pred, test_data.y)
    #         print 'epoch: {:>3}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
    #             i, test_loss, p * 100, r * 100, f1 * 100
    #         )

    # # rnn test
    # lstm_setting = LSTMSetting()
    # lstm_model = LSTM(data_loader.wordembedding, lstm_setting)
    #
    # with tf.Session() as session:
    #     tf.global_variables_initializer().run()
    #     test_data = data_loader.get_test_data()
    #     for i in range(100):
    #         j = 0
    #         batches = data_loader.get_train_batches(batch_size=1024)
    #         for batch in batches:
    #             j += 1
    #             loss = lstm_model.fit(session, batch, dropout_keep_rate=0.5)
    #             _, c_label = lstm_model.predict(session, batch)
    #             if j % 100 == 0:
    #                 p, r, f1 = get_p_r_f1(c_label, batch.y)
    #                 print 'epoch: {:>3}, batch: {:>4}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
    #                     i, j, loss, p * 100, r * 100, f1 * 100
    #                 )
    #         test_loss, test_pred = lstm_model.predict(session, test_data)
    #         p, r, f1 = get_p_r_f1(test_pred, test_data.y)
    #         print 'epoch: {:>3}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
    #             i, test_loss, p * 100, r * 100, f1 * 100
    #         )

    # bigru test
    bigru_setting = BiGRUSetting()
    bigru_model = BiGRU(data_loader.wordembedding, bigru_setting)

    # with tf.Session() as session:
    #     tf.global_variables_initializer().run()
    #     for i in range(100):
    #         batches = data_loader.get_train_batches(batch_size=128)
    #         for batch in batches:
    #             feed_dict = {bigru_model.input_words: batch.word,
    #                          bigru_model.input_pos1: batch.pos1,
    #                          bigru_model.input_pos2: batch.pos2,
    #                          bigru_model.input_labels: batch.y,
    #                          bigru_model.dropout_keep_rate: 0.5
    #                          }
    #             op, opf, opb = session.run([bigru_model.outputs], feed_dict=feed_dict)
    #             pass
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        test_data = data_loader.get_test_data()
        for i in range(100):
            j = 0
            batches = data_loader.get_train_batches(batch_size=1024)
            for batch in batches:
                j += 1
                loss = bigru_model.fit(session, batch, dropout_keep_rate=0.5)
                _, c_label = bigru_model.predict(session, batch)
                if j % 100 == 0:
                    p, r, f1 = get_p_r_f1(c_label, batch.y)
                    print 'epoch: {:>3}, batch: {:>4}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
                        i, j, loss, p * 100, r * 100, f1 * 100
                    )
            test_loss, test_pred = bigru_model.predict(session, test_data)
            p, r, f1 = get_p_r_f1(test_pred, test_data.y)
            print 'epoch: {:>3}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
                i, test_loss, p * 100, r * 100, f1 * 100
            )

if __name__ == '__main__':
    main()
