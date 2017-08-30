# -*- encoding: utf-8 -*-
# Created by han on 17-7-8

import os
import cPickle
from datetime import datetime
from data_loader import DataLoader
from model_settings import *
from models import *
from evaluate import *

ms_dict = {
    'cnn': CNNSetting,
    'rnn': RNNSetting,
    'birnn': RNNSetting,
    'birnn_att': RNNSetting,
    'rnn_mi': RNNMiSetting
}
m_dict = {
    'cnn': CNN,
    'rnn': RNN,
    'birnn': BiRNN,
    'birnn_att': BiRNN_ATT,
    'rnn_mi': RNN_MI
}


def main():
    model_name = 'rnn_mi'
    train_epochs_num = 100
    batch_size = 128

    with open('./data/id2word.pkl', 'rb') as f:
        id2word = cPickle.load(f)
    with open('./data/word2id.pkl', 'rb') as f:
        word2id = cPickle.load(f)
    with open('./origin_data/idx2rel.pkl', 'rb') as f:
        id2rel = cPickle.load(f)

    res_path = os.path.join('./result', model_name)
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    res_prf = os.path.join(res_path, 'prf.txt')
    res_ana = os.path.join(res_path, 'analysis.txt')
    log_prf = tf.gfile.GFile(res_prf, mode='a')
    log_ana = tf.gfile.GFile(res_ana, mode='a')

    best_f1 = 0
    if '_mi' not in model_name:
        model_setting = ms_dict[model_name]()
        if 'cnn' in model_name:
            data_loader = DataLoader('./data', multi_ins=False, cnn_win_size=model_setting.win_size)
        else:
            data_loader = DataLoader('./data', multi_ins=False)
        model = m_dict[model_name](data_loader.wordembedding, model_setting)
        with tf.Session() as session:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(max_to_keep=None)
            test_data = data_loader.get_test_data()
            for epoch_num in range(train_epochs_num):
                # train
                iter_num = 0
                batches = data_loader.get_train_batches(batch_size=batch_size)
                for batch in batches:
                    iter_num += 1
                    loss = model.fit(session, batch, dropout_keep_rate=model_setting.dropout_rate)
                    _, c_label, _ = model.evaluate(session, batch)
                    if iter_num % 100 == 0:
                        _, prf_macro, _ = get_p_r_f1(c_label, batch.y)
                        p, r, f1 = prf_macro
                        log_info = 'train: ' + str(datetime.now()) + 'epoch: {:>3}, batch: {:>4}, lost: {:.3f}, ' \
                                                                     'p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%\n'.format(
                            epoch_num, iter_num, loss, p * 100, r * 100, f1 * 100
                        )
                        log_prf.write(log_info)
                        print 'train: ', datetime.now(), 'epoch: {:>3}, batch: {:>4}, lost: {:.3f}, p: {:.3f}%, ' \
                                                         'r: {:.3f}%, f1:{:.3f}%'.format(
                            epoch_num, iter_num, loss, p * 100, r * 100, f1 * 100
                        )
                # test
                use_neg = True
                test_loss, test_pred, test_prob = model.evaluate(session, test_data)
                prf_list, prf_macro, prf_micro = get_p_r_f1(test_pred, test_data.y, use_neg)
                p, r, f1 = prf_macro
                log_info = 'test: ' + str(datetime.now()) + ' epoch: {:>3}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%,' \
                                                            ' f1:{:.3f}%\n'.format(
                    epoch_num, test_loss, p * 100, r * 100, f1 * 100
                )
                if f1 > best_f1:
                    best_f1 = f1
                    # record p, r, f1
                    log_ana.write(log_info)
                    if use_neg:
                        for id in range(len(prf_list)):
                            rel_prf = 'rel: {:>2}_{:<6}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%\n'.format(
                                id, id2rel[id], prf_list[id][3] * 100, prf_list[id][4] * 100, prf_list[id][5] * 100
                            )
                            log_ana.write(rel_prf)
                    else:
                        for id in range(len(prf_list)):
                            rel_prf = 'rel: {}_{}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%\n'.format(
                                id+1, id2rel[id+1], prf_list[id][3] * 100, prf_list[id][4] * 100, prf_list[id][5] * 100
                            )
                            log_ana.write(rel_prf)
                    # record wrong instance
                    wrong_ins = get_wrong_ins(test_pred, test_data, word2id, id2word, id2rel, use_neg)
                    wrong_ins = sorted(wrong_ins, key=lambda x: x[1])
                    wrong_ins = ['\t'.join(i) + '\n' for i in wrong_ins]
                    for ins in wrong_ins:
                        log_ana.write(ins)
                    log_ana.write('-' * 80 + '\n')
                    # draw pr curve
                    prc_fn = os.path.join(res_path, 'prc_epoch{}.png'.format(epoch_num))
                    save_prcurve(test_prob, test_data.y, model_name, prc_fn)
                    saver.save(session, os.path.join(res_path, 'model_saved'))

                log_prf.write(log_info)
                print 'test: ', datetime.now(), 'epoch: {:>3}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
                    epoch_num, test_loss, p * 100, r * 100, f1 * 100
                )
    else:
        data_loader = DataLoader('./data', multi_ins=True)
        batch_size = 128
        print 'building {} model...'.format(model_name)
        rnn_mi_setting = RNNMiSetting()
        rnn_mi_setting.bag_num = batch_size
        rnn_mi_model = RNN_MI(data_loader.wordembedding, rnn_mi_setting)

        with tf.Session() as session:
            tf.global_variables_initializer().run()
            for epoch_num in range(train_epochs_num):
                iter_num = 0
                train_batches = data_loader.get_train_batches(batch_size=batch_size)
                test_batches = data_loader.get_test_batches(batch_size=batch_size, use_single=True)
                for batch in train_batches:
                    iter_num += 1
                    acc, loss = rnn_mi_model.fit(session, batch, dropout_keep_rate=0.5)
                    acc = np.mean(np.reshape(np.array(acc), batch_size))
                    if iter_num % 100 == 0:
                        print datetime.now(), 'epoch: {:>3}, batch: {:>4}, lost: {:.3f}, acc: {:.3f}'.format(
                            epoch_num, iter_num, loss, acc
                        )
                test_loss, test_pred, test_label = [], [], []
                for batch in test_batches:
                    batch_loss, batch_pred = rnn_mi_model.evaluate(session, batch)
                    test_loss.append(batch_loss)
                    test_pred += batch_pred
                    test_label += list(batch.y)
                test_loss = np.mean(test_loss)
                prf_list, prf_macro, prf_micro = get_p_r_f1(test_pred, test_label.y)
                p, r, f1 = prf_macro
                print datetime.now(), 'epoch: {:>3}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
                    epoch_num, test_loss, p * 100, r * 100, f1 * 100
                )


if __name__ == '__main__':
    main()
