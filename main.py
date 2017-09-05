# -*- encoding: utf-8 -*-
# Created by han on 17-7-8

import os
import cPickle
import time
from datetime import datetime
from data_loader import DataLoader
from model_settings import *
from models import *
from evaluate import *

ms_dict = {
    'cnn': CnnSetting,
    'rnn': RnnSetting,
    'birnn': RnnSetting,
    'birnn_att': RnnSetting,
    'birnn_selfatt': RnnSetting_SelfAtt,
    'birnn_mi': RnnMiSetting
}

m_dict = {
    'cnn': Cnn,
    'rnn': Rnn,
    'birnn': BiRnn,
    'birnn_att': BiRnn_Att,
    'birnn_selfatt': BiRnn_SelfAtt,
    'birnn_mi': BiRnn_Mi
}


def get_ids(c_feature):
    """
    get index of word, character and relation
    :param c_feature: if c_feature is true, use character level features
    :return:
    """
    if c_feature:
        with open('./data/id2char.pkl', 'rb') as f:
            id2x = cPickle.load(f)
        with open('./data/char2id.pkl', 'rb') as f:
            x2id = cPickle.load(f)
    else:
        with open('./data/id2word.pkl', 'rb') as f:
            id2x = cPickle.load(f)
        with open('./data/word2id.pkl', 'rb') as f:
            x2id = cPickle.load(f)
    with open('./origin_data/idx2rel.pkl', 'rb') as f:
        id2rel = cPickle.load(f)

    return x2id, id2x, id2rel


def train_evaluate(data_loader, model, model_setting, train_epochs_num, batch_size):
    """
    train and evaluate model
    """
    # get indexes
    x2id, id2x, id2rel = get_ids(data_loader.c_feature)

    # result saving path
    res_path = os.path.join(
        './result', model.model_name, '{}_{}_{}'.format(
            model.model_name,
            'c' if data_loader.c_feature else 'w',
            time.strftime('%y%m%d_%H%M', time.localtime(time.time()))
        )
    )
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    # log files
    log_prf = tf.gfile.GFile(os.path.join(res_path, 'prf.txt'), mode='a')
    log_ana = tf.gfile.GFile(os.path.join(res_path, 'analysis.txt'), mode='a')
    log_prf.write('=' * 80 + '\n' + model.model_name + '\n')
    log_ana.write('=' * 80 + '\n' + model.model_name + '\n')
    for para in sorted(model_setting.__dict__.keys()):
        pv_str = '{}: {}'.format(para, getattr(model_setting, para))
        print pv_str
        log_prf.write(pv_str + '\n')
        log_ana.write(pv_str + '\n')
    log_prf.write('=' * 80 + '\n')
    log_ana.write('=' * 80 + '\n')

    with tf.Session() as session:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(max_to_keep=None)
        # best evaluation f1
        best_test_f1 = 0
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
                    log_info = 'train: ' + str(datetime.now()) + ' epoch: {:>3}, batch: {:>4}, lost: {:.3f},' \
                                                                 ' p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%\n'.format(
                        epoch_num, iter_num, loss, p * 100, r * 100, f1 * 100
                    )
                    log_prf.write(log_info)
                    print 'train: ', datetime.now(), ' epoch: {:>3}, batch: {:>4}, lost: {:.3f}, p: {:.3f}%,' \
                                                     ' r: {:.3f}%, f1:{:.3f}%'.format(
                        epoch_num, iter_num, loss, p * 100, r * 100, f1 * 100
                    )

            # test
            use_neg = True
            test_batches = data_loader.get_test_batches(batch_size)
            test_loss, test_prob, test_pred, test_ans = [], [], [], []
            test_x, test_p1, test_p2 = [], [], []
            for batch in test_batches:
                batch_loss, batch_pred, batch_prob = model.evaluate(session, batch)
                test_loss.append(batch_loss)
                test_prob += list(batch_prob)
                test_pred += list(batch_pred)
                test_ans += list(batch.y)
                if not data_loader.multi_ins:
                    test_x += list(batch.x)
                    test_p1 += list(batch.pos1[:, 0])
                    test_p2 += list(batch.pos2[:, 0])
            test_loss = np.mean(test_loss)
            prf_list, prf_macro, prf_micro = get_p_r_f1(test_pred, test_ans, use_neg)
            p, r, f1 = prf_macro
            log_info = 'test : ' + str(datetime.now()) + ' epoch: {:>3}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%,' \
                                                         ' f1:{:.3f}%\n'.format(
                epoch_num, test_loss, p * 100, r * 100, f1 * 100
            )

            # best performance
            if f1 > best_test_f1:
                best_test_f1 = f1
                # record p, r, f1
                log_ana.write(log_info)
                if use_neg:
                    for idx in range(len(prf_list)):
                        rel_prf = 'rel: {:>2}_{:<6}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%\n'.format(
                            idx, id2rel[idx], prf_list[idx][3] * 100, prf_list[idx][4] * 100, prf_list[idx][5] * 100
                        )
                        log_ana.write(rel_prf)
                else:
                    for idx in range(len(prf_list)):
                        rel_prf = 'rel: {}_{}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%\n'.format(
                            idx+1, id2rel[idx+1],
                            prf_list[idx][3] * 100, prf_list[idx][4] * 100, prf_list[idx][5] * 100
                        )
                        log_ana.write(rel_prf)

                # record wrong instance multi-instance do no support this process
                if not data_loader.multi_ins:
                    wrong_ins = get_wrong_ins(
                        test_pred, test_ans, test_x, test_p1, test_p2, x2id, id2x, id2rel, use_neg
                    )
                    wrong_ins = sorted(wrong_ins, key=lambda x: x[1])
                    wrong_ins = ['\t'.join(i) + '\n' for i in wrong_ins]
                    for ins in wrong_ins:
                        log_ana.write(ins)
                log_ana.write('-' * 80 + '\n')

                # draw pr curve
                # prc_fn = os.path.join(res_path, 'prc_epoch{}.png'.format(epoch_num))
                # save_prcurve(test_prob, test_ans, model_name, prc_fn)

                # save model
                saver.save(session, os.path.join(res_path, 'model_saved'), epoch_num)

            log_prf.write(log_info)
            print 'test : ', datetime.now(), ' epoch: {:>3}, lost: {:.3f}, p: {:.3f}%, r: {:.3f}%, f1:{:.3f}%'.format(
                epoch_num, test_loss, p * 100, r * 100, f1 * 100
            )


def main():
    """
    initialize, train and evaluate models
    """
    model_name = 'birnn_mi'

    # feature select
    c_feature = False

    # train and test parameters
    train_epochs_num = 200
    batch_size = 1024

    # model setting
    model_setting = ms_dict[model_name]()

    # initialize data loader
    print 'data loader initializing...'
    mult_ins = True if '_mi' in model_name else False

    if 'cnn' in model_name:
        data_loader = DataLoader(
            './data', c_feature=c_feature, multi_ins=mult_ins, cnn_win_size=model_setting.win_size
        )
    else:
        data_loader = DataLoader('./data', c_feature=c_feature, multi_ins=mult_ins)

    # update model setting
    model_setting.sen_len = data_loader.max_sen_len
    if mult_ins:
        model_setting.bag_num = batch_size

    # initialize model
    print 'initializing {} model...'.format(model_name)
    model = m_dict[model_name](data_loader.embedding, model_setting)

    # train and evaluate
    print 'training and evaluating model...'
    train_evaluate(data_loader, model, model_setting, train_epochs_num, batch_size)


if __name__ == '__main__':
    main()
