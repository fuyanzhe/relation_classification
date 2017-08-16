# -*- encoding: utf-8 -*-
# Created by han on 17-7-8

from data_loader import DataLoader
from model_settings import *
from models import *
from evaluate import *
from datetime import datetime


def main():
    model_name = 'birnn_att'
    train_epochs_num = 100

    if model_name == 'cnn':
        """
        epoch:   6, lost: 0.614, p: 81.678%, r: 89.788%, f1:84.531%
        """
        # cnn test
        data_loader = DataLoader('./data', multi_ins=False)
        print 'building {} model...'.format(model_name)
        cnn_setting = CNNSetting()
        cnn_model = CNN(data_loader.wordembedding, cnn_setting)

        with tf.Session() as session:
            tf.global_variables_initializer().run()
            test_data = data_loader.get_test_data()
            for epoch_num in range(train_epochs_num):
                iter_num = 0
                batches = data_loader.get_train_batches(batch_size=128)
                for batch in batches:
                    iter_num += 1
                    loss = cnn_model.fit(session, batch, dropout_keep_rate=0.5)
                    _, c_label = cnn_model.evaluate(session, batch)
                    if iter_num % 100 == 0:
                        p, r, f1 = get_p_r_f1(c_label, batch.y)
                        print datetime.now(), 'epoch: {:>3}, batch: {:>4}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
                            epoch_num, iter_num, loss, p * 100, r * 100, f1 * 100
                        )
                test_loss, test_pred = cnn_model.evaluate(session, test_data)
                p, r, f1 = get_p_r_f1(test_pred, test_data.y)
                print datetime.now(), 'epoch: {:>3}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
                    epoch_num, test_loss, p * 100, r * 100, f1 * 100
                )
    elif model_name == 'rnn':
        """
        epoch:   2, lost: 0.619, p: 81.377%, r: 90.501%, f1:84.603%
        """
        # rnn test
        data_loader = DataLoader('./data', multi_ins=False)
        print 'building {} model...'.format(model_name)
        rnn_setting = RNNSetting()
        rnn_model = RNN(data_loader.wordembedding, rnn_setting)

        with tf.Session() as session:
            tf.global_variables_initializer().run()
            test_data = data_loader.get_test_data()
            for epoch_num in range(train_epochs_num):
                iter_num = 0
                batches = data_loader.get_train_batches(batch_size=128)
                for batch in batches:
                    iter_num += 1
                    loss = rnn_model.fit(session, batch, dropout_keep_rate=0.5)
                    _, c_label = rnn_model.evaluate(session, batch)
                    if iter_num % 100 == 0:
                        p, r, f1 = get_p_r_f1(c_label, batch.y)
                        print datetime.now(), 'epoch: {:>3}, batch: {:>4}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
                            epoch_num, iter_num, loss, p * 100, r * 100, f1 * 100
                        )
                test_loss, test_pred = rnn_model.evaluate(session, test_data)
                p, r, f1 = get_p_r_f1(test_pred, test_data.y)
                print datetime.now(), ' epoch: {:>3}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
                    epoch_num, test_loss, p * 100, r * 100, f1 * 100
                )
    elif model_name == 'birnn':
        """
        epoch: 2, p:82.245, r:90.998, f:85.222
        """

        # bigru test
        data_loader = DataLoader('./data', multi_ins=False)
        print 'building {} model...'.format(model_name)
        birnn_setting = RNNSetting()
        birnn_model = BiRNN(data_loader.wordembedding, birnn_setting)

        with tf.Session() as session:
            tf.global_variables_initializer().run()
            test_data = data_loader.get_test_data()
            for epoch_num in range(train_epochs_num):
                iter_num = 0
                batches = data_loader.get_train_batches(batch_size=128)
                for batch in batches:
                    iter_num += 1
                    loss = birnn_model.fit(session, batch, dropout_keep_rate=0.5)
                    _, c_label = birnn_model.evaluate(session, batch)
                    if iter_num % 100 == 0:
                        p, r, f1 = get_p_r_f1(c_label, batch.y)
                        print datetime.now(), 'epoch: {:>3}, batch: {:>4}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
                            epoch_num, iter_num, loss, p * 100, r * 100, f1 * 100
                        )
                test_loss, test_pred = birnn_model.evaluate(session, test_data)
                p, r, f1 = get_p_r_f1(test_pred, test_data.y)
                print datetime.now(), 'epoch: {:>3}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
                    epoch_num, test_loss, p * 100, r * 100, f1 * 100
                )
    elif model_name == 'birnn_att':
        # bigru_att test
        data_loader = DataLoader('./data', multi_ins=False)
        print 'building {} model...'.format(model_name)
        rnn_att_setting = RNNSetting()
        rnn_att_model = BiRNN_ATT(data_loader.wordembedding, rnn_att_setting)

        with tf.Session() as session:
            tf.global_variables_initializer().run()
            test_data = data_loader.get_test_data()
            for epoch_num in range(train_epochs_num):
                iter_num = 0
                batches = data_loader.get_train_batches(batch_size=128)
                for batch in batches:
                    iter_num += 1
                    loss = rnn_att_model.fit(session, batch, dropout_keep_rate=0.5)
                    _, c_label = rnn_att_model.evaluate(session, batch)
                    if iter_num % 100 == 0:
                        p, r, f1 = get_p_r_f1(c_label, batch.y)
                        print datetime.now(), 'epoch: {:>3}, batch: {:>4}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
                            epoch_num, iter_num, loss, p * 100, r * 100, f1 * 100
                        )
                test_loss, test_pred = rnn_att_model.evaluate(session, test_data)
                p, r, f1 = get_p_r_f1(test_pred, test_data.y)
                print datetime.now(), 'epoch: {:>3}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
                    epoch_num, test_loss, p * 100, r * 100, f1 * 100
                )
    elif model_name == 'rnn_mi':
        # bigru _mi test
        data_loader = DataLoader('./data', multi_ins=True)
        batch_size = 128
        print 'building {} model...'.format(model_name)
        rnn_mi_setting = RNNMiSetting()
        rnn_mi_setting.bag_num = batch_size
        rnn_mi_model = RNN_MI(data_loader.wordembedding, rnn_mi_setting)

        with tf.Session() as session:
            tf.global_variables_initializer().run()
            _, test_data = data_loader.get_test_data()
            for epoch_num in range(train_epochs_num):
                iter_num = 0
                batches = data_loader.get_train_batches(batch_size=batch_size)
                for batch in batches:
                    iter_num += 1
                    loss = rnn_mi_model.fit(session, batch, dropout_keep_rate=0.5)
                    # _, c_label = rnn_mi_model.evaluate(session, batch)
                    if iter_num % 100 == 0:
                        # p, r, f1 = get_p_r_f1(c_label, batch.y)
                        # print datetime.now(), 'epoch: {:>3}, batch: {:>4}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
                        #     epoch_num, iter_num, loss, p * 100, r * 100, f1 * 100
                        # )
                        print datetime.now(), 'epoch: {:>3}, batch: {:>4}, lost: {:.3f}'.format(
                            epoch_num, iter_num, loss
                        )
                test_loss, test_pred = rnn_mi_model.evaluate(session, test_data)
                p, r, f1 = get_p_r_f1(test_pred, test_data.y)
                print datetime.now(), 'epoch: {:>3}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
                    epoch_num, test_loss, p * 100, r * 100, f1 * 100
                )


if __name__ == '__main__':
    main()
