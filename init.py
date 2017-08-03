# -*- encoding: utf-8 -*-
# Created by han on 17-6-18
import numpy as np
import pandas as pd
import cPickle


# embedding the position
def pos_embed(x, max_len):
    if x < -max_len:
        return 0
    if -max_len <= x <= max_len:
        return x + max_len + 1
    if x > max_len:
        return 2 * max_len + 2


# find the index of x in y, if x not in y, return -1
def find_index(x, y):
    flag = -1
    for i in range(len(y)):
        if x != y[i]:
            continue
        else:
            return i
    return flag


# reading data
def init():
    print 'reading word embedding data...'
    vec = []
    word2id = {}
    with open('./origin_data/vectors/word_vec_50.pkl', 'rb') as f:
        word2vec = cPickle.load(f)
    for k in sorted(word2vec.keys()):
        word2id[k] = len(word2id)
        vec.append(word2vec[k])

    word2id[u'UNK'] = len(word2id)
    word2id[u'BLANK'] = len(word2id)

    dim = 50
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec = np.array(vec, dtype=np.float32)
    # save vec matrix
    np.save('./data/vec.npy', vec)

    print 'reading relation to id'
    with open('./origin_data/rel2idx.pkl', 'rb') as f:
        relation2id = cPickle.load(f)

    # length of sentence is 100
    fixlen = 100
    # max length of position embedding is 90 (-90~+90)
    maxlen = 90

    # train data
    # single-instance
    train_sen_all = []
    train_sen_pos1_all = []
    train_sen_pos2_all = []
    train_sen_len_all = []
    train_label_all = []

    # multi-instance
    # {entity pair:[[[label1-sentence 1],[label1-sentence 2]...],[[label2-sentence 1],[label2-sentence 2]...]}
    train_sen = {}
    # {entity pair:[[label1-sentence 1 length, label1-sentence 2 length],...]}
    train_sen_len = {}
    # {entity pair:[label1,label2,...]} the label is one-hot vector
    train_ans = {}

    print 'reading train data...'

    data_train = pd.HDFStore('./origin_data/instances_rel_train.h5')
    for rel in sorted(int(i.replace('/', '')) for i in data_train.keys()):
        data_rel = data_train['/' + str(rel)]
        for _, ins in data_rel.iterrows():
            # get entity name
            en1 = ins['uri1']
            en2 = ins['uri2']
            relation = ins['rel']
            if rel != 0:
                assert rel == relation2id[relation]

            # put the same entity pair sentences into a dict
            tup = (en1, en2)

            # label one-hot
            y_id = rel
            label = [0] * len(relation2id)
            label[y_id] = 1

            # single-instance
            train_label_all.append(label)

            # multi-instance
            if tup not in train_sen:
                train_sen[tup] = []
                train_sen[tup].append([])
                train_sen_len[tup] = []
                train_sen_len[tup].append([])
                label_tag = 0
                train_ans[tup] = []
                train_ans[tup].append(label)
            else:
                temp = find_index(label, train_ans[tup])
                if temp == -1:
                    train_ans[tup].append(label)
                    label_tag = len(train_ans[tup]) - 1
                    train_sen[tup].append([])
                    train_sen_len[tup].append([])
                else:
                    label_tag = temp

            # get sentence
            sentence = ins['st_seg']
            sen_len = len(sentence)

            # get entity pos
            en1pos = ins['ent1_p2']
            en2pos = ins['ent2_p2']

            output = []
            # get relative position
            for i in range(fixlen):
                word = word2id[u'BLANK']
                rel_e1 = pos_embed(i - en1pos, maxlen)
                rel_e2 = pos_embed(i - en2pos, maxlen)
                output.append([word, rel_e1, rel_e2])

            # translate the words in sentences to index
            for i in range(min(fixlen, len(sentence))):
                if sentence[i].decode('utf8') not in word2id:
                    word = word2id[u'UNK']
                else:
                    word = word2id[sentence[i].decode('utf8')]
                output[i][0] = word

            # single-instance
            sen_idx = [i[0] for i in output]
            sen_p1 = [i[1] for i in output]
            sen_p2 = [i[2] for i in output]
            train_sen_all.append(sen_idx)
            train_sen_pos1_all.append(sen_p1)
            train_sen_pos2_all.append(sen_p2)
            train_sen_len_all.append(sen_len)
            # multi-instance
            train_sen[tup][label_tag].append(output)
            train_sen_len[tup][label_tag].append(sen_len)

    data_train.close()

    # test data
    # single-instance
    test_sen_all = []
    test_sen_pos1_all = []
    test_sen_pos2_all = []
    test_sen_len_all = []
    test_label_all = []

    # multi-instance
    # {entity pair:[[sentence 1],[sentence 2]...]}
    test_sen = {}
    # {entity pair:[sentence 1 length, sentence 2 length],...}
    test_sen_len = {}
    # {entity pair:[labels,...]} the labels is N-hot vector (N is the number of multi-label)
    test_ans = {}

    print('reading test data ...')

    data_test = pd.HDFStore('./origin_data/instances_rel_test.h5')
    for rel in sorted(int(i.replace('/', '')) for i in data_test.keys()):
        data_rel = data_test['/' + str(rel)]
        for _, ins in data_rel.iterrows():
            # get entity name
            en1 = ins['uri1']
            en2 = ins['uri2']
            relation = ins['rel']
            if rel != 0:
                assert rel == relation2id[relation]

            # put the same entity pair sentences into a dict
            tup = (en1, en2)

            y_id = rel
            label = [0] * len(relation2id)
            label[y_id] = 1

            # single-instance
            test_label_all.append(label)

            # multi-instance
            if tup not in test_sen:
                test_sen[tup] = []
                test_sen_len[tup] = []
                test_ans[tup] = label
            else:
                test_ans[tup][y_id] = 1

            # get sentence
            sentence = ins['st_seg']
            sen_len = len(sentence)

            # get entity pos
            en1pos = ins['ent1_p2']
            en2pos = ins['ent2_p2']

            output = []
            # pos feature
            for i in range(fixlen):
                word = word2id[u'BLANK']
                rel_e1 = pos_embed(i - en1pos, maxlen)
                rel_e2 = pos_embed(i - en2pos, maxlen)
                output.append([word, rel_e1, rel_e2])

            for i in range(min(fixlen, len(sentence))):
                if sentence[i].decode('utf8') not in word2id:
                    word = word2id[u'UNK']
                else:
                    word = word2id[sentence[i].decode('utf8')]

                output[i][0] = word

            # single-instance
            sen_idx = [i[0] for i in output]
            sen_p1 = [i[1] for i in output]
            sen_p2 = [i[2] for i in output]
            test_sen_all.append(sen_idx)
            test_sen_pos1_all.append(sen_p1)
            test_sen_pos2_all.append(sen_p2)
            test_sen_len_all.append(sen_len)
            # multi-instance
            test_sen[tup].append(output)
            test_sen_len[tup].append(sen_len)

    data_test.close()

    # single-instance
    np.save('./data/s-ins/single_train_word_.npy', np.array(train_sen_all))
    np.save('./data/s-ins/single_train_pos1_.npy', np.array(train_sen_pos1_all))
    np.save('./data/s-ins/single_train_pos2_.npy', np.array(train_sen_pos2_all))
    np.save('./data/s-ins/single_train_len.npy', np.array(train_sen_len_all))
    np.save('./data/s-ins/single_train_y.npy', np.array(train_label_all))

    np.save('./data/s-ins/single_test_word.npy', np.array(test_sen_all))
    np.save('./data/s-ins/single_test_pos1.npy', np.array(test_sen_pos1_all))
    np.save('./data/s-ins/single_test_pos2.npy', np.array(test_sen_pos2_all))
    np.save('./data/s-ins/single_test_len.npy', np.array(test_sen_len_all))
    np.save('./data/s-ins/single_test_y.npy', np.array(test_label_all))

    # multi-instance
    train_x = []
    train_x_len = []
    train_y = []
    test_x = []
    test_x_len = []
    test_y = []

    print 'organizing train data'
    f = open('./data/m-ins/train_q&a.txt', 'w')
    temp = 0
    for i in train_sen:
        assert len(train_ans[i]) == len(train_sen[i])
        # label number
        lenth = len(train_ans[i])
        for j in range(lenth):
            train_x.append(train_sen[i][j])
            train_x_len.append(train_sen_len[i][j])
            train_y.append(train_ans[i][j])
            f.write(str(temp) + '\t' + i[0] + '\t' + i[1] + '\t' + str(np.argmax(train_ans[i][j])) + '\n')
            temp += 1
    f.close()

    print 'organizing test data'
    f = open('./data/m-ins/test_q&a.txt', 'w')
    temp = 0
    for i in test_sen:
        test_x.append(test_sen[i])
        test_x_len.append(test_sen_len[i])
        test_y.append(test_ans[i])
        tempstr = ''
        for j in range(len(test_ans[i])):
            if test_ans[i][j] != 0:
                tempstr = tempstr + str(j) + '\t'
        f.write(str(temp) + '\t' + i[0] + '\t' + i[1] + '\t' + str(tempstr) + '\n')
        temp += 1
    f.close()

    train_x = np.array(train_x)
    train_x_len = np.array(train_x_len)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_x_len = np.array(test_x_len)
    test_y = np.array(test_y)

    np.save('./data/m-ins/train_x.npy', train_x)
    np.save('./data/m-ins/train_len.npy', train_x_len)
    np.save('./data/m-ins/train_y.npy', train_y)
    np.save('./data/m-ins/testall_x.npy', test_x)
    np.save('./data/m-ins/testall_len.npy', test_x_len)
    np.save('./data/m-ins/testall_y.npy', test_y)

    # get test data for P@N evaluation, in which only entity pairs with more than 1 sentence exist
    print 'get test data for p@n test'

    pone_test_x = []
    pone_test_x_len = []
    pone_test_y = []

    ptwo_test_x = []
    ptwo_test_x_len = []
    ptwo_test_y = []

    pall_test_x = []
    pall_test_x_len = []
    pall_test_y = []

    for i in range(len(test_x)):
        if len(test_x[i]) > 1:

            pall_test_x.append(test_x[i])
            pall_test_x_len.append(test_x_len[i])
            pall_test_y.append(test_y[i])

            onetest = []
            onetest_len = []
            temp = np.random.randint(len(test_x[i]))
            onetest.append(test_x[i][temp])
            onetest_len.append(test_x_len[i][temp])
            pone_test_x.append(onetest)
            pone_test_x_len.append(onetest_len)
            pone_test_y.append(test_y[i])

            twotest = []
            twotest_len = []
            temp1 = np.random.randint(len(test_x[i]))
            temp2 = np.random.randint(len(test_x[i]))
            while temp1 == temp2:
                temp2 = np.random.randint(len(test_x[i]))
            twotest.append(test_x[i][temp1])
            twotest_len.append(test_x_len[i][temp1])
            twotest.append(test_x[i][temp2])
            ptwo_test_x.append(twotest)
            ptwo_test_x_len.append(twotest_len)
            ptwo_test_y.append(test_y[i])

    pone_test_x = np.array(pone_test_x)
    pone_test_x_len = np.array(pone_test_x_len)
    pone_test_y = np.array(pone_test_y)
    ptwo_test_x = np.array(ptwo_test_x)
    ptwo_test_x_len = np.array(ptwo_test_x_len)
    ptwo_test_y = np.array(ptwo_test_y)
    pall_test_x = np.array(pall_test_x)
    pall_test_x_len = np.array(pall_test_x_len)
    pall_test_y = np.array(pall_test_y)

    np.save('./data/m-ins/pone_test_x.npy', pone_test_x)
    np.save('./data/m-ins/pone_test_len.npy', pone_test_x_len)
    np.save('./data/m-ins/pone_test_y.npy', pone_test_y)
    np.save('./data/m-ins/ptwo_test_x.npy', ptwo_test_x)
    np.save('./data/m-ins/ptwo_test_len.npy', ptwo_test_x_len)
    np.save('./data/m-ins/ptwo_test_y.npy', ptwo_test_y)
    np.save('./data/m-ins/pall_test_x.npy', pall_test_x)
    np.save('./data/m-ins/pall_test_len.npy', pall_test_x_len)
    np.save('./data/m-ins/pall_test_y.npy', pall_test_y)


def seperate():
    # train
    print 'reading training data'
    x_train = np.load('./data/m-ins/train_x.npy')

    train_word = []
    train_pos1 = []
    train_pos2 = []

    print 'seprating train data'
    for tup_bag in x_train:
        word = []
        pos1 = []
        pos2 = []
        for sent in tup_bag:
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            for word_feature in sent:
                temp_word.append(word_feature[0])
                temp_pos1.append(word_feature[1])
                temp_pos2.append(word_feature[2])
            word.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        train_word.append(word)
        train_pos1.append(pos1)
        train_pos2.append(pos2)

    train_word = np.array(train_word)
    train_pos1 = np.array(train_pos1)
    train_pos2 = np.array(train_pos2)
    np.save('./data/m-ins/train_word.npy', train_word)
    np.save('./data/m-ins/train_pos1.npy', train_pos1)
    np.save('./data/m-ins/train_pos2.npy', train_pos2)

    # test
    print 'seperating test all data'
    x_test = np.load('./data/m-ins/testall_x.npy')

    test_word = []
    test_pos1 = []
    test_pos2 = []

    for tup_bag in x_test:
        word = []
        pos1 = []
        pos2 = []
        for sent in tup_bag:
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            for word_feature in sent:
                temp_word.append(word_feature[0])
                temp_pos1.append(word_feature[1])
                temp_pos2.append(word_feature[2])
            word.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        test_word.append(word)
        test_pos1.append(pos1)
        test_pos2.append(pos2)

    test_word = np.array(test_word)
    test_pos1 = np.array(test_pos1)
    test_pos2 = np.array(test_pos2)

    np.save('./data/m-ins/testall_word.npy', test_word)
    np.save('./data/m-ins/testall_pos1.npy', test_pos1)
    np.save('./data/m-ins/testall_pos2.npy', test_pos2)

    for pn in ['pone', 'ptwo', 'pall']:
        print 'reading {} test data'.format(pn)
        x_test = np.load('./data/m-ins/{}_test_x.npy'.format(pn))

        test_word = []
        test_pos1 = []
        test_pos2 = []

        for tup_bag in x_test:
            word = []
            pos1 = []
            pos2 = []
            for sent in tup_bag:
                temp_word = []
                temp_pos1 = []
                temp_pos2 = []
                for word_feature in sent:
                    temp_word.append(word_feature[0])
                    temp_pos1.append(word_feature[1])
                    temp_pos2.append(word_feature[2])
                word.append(temp_word)
                pos1.append(temp_pos1)
                pos2.append(temp_pos2)
            test_word.append(word)
            test_pos1.append(pos1)
            test_pos2.append(pos2)

        test_word = np.array(test_word)
        test_pos1 = np.array(test_pos1)
        test_pos2 = np.array(test_pos2)
        np.save('./data/m-ins/{}_test_word.npy'.format(pn), test_word)
        np.save('./data/m-ins/{}_test_pos1.npy'.format(pn), test_pos1)
        np.save('./data/m-ins/{}_test_pos2.npy'.format(pn), test_pos2)


def getsmall():
    print 'reading training data'
    word = np.load('./data/m-ins/train_word.npy')
    pos1 = np.load('./data/m-ins/train_pos1.npy')
    pos2 = np.load('./data/m-ins/train_pos2.npy')
    slen = np.load('./data/m-ins/train_len.npy')
    y = np.load('./data/m-ins/train_y.npy')

    new_word = []
    new_pos1 = []
    new_pos2 = []
    new_slen = []
    new_y = []

    # we slice some big batch in train data into small batches in case of running out of memory
    print 'get small training data'
    c0, c1, c2, c3, c4 = 0, 0, 0, 0, 0
    for i in range(len(word)):
        lenth = len(word[i])
        if lenth <= 1000:
            c0 += 1
            new_word.append(word[i])
            new_pos1.append(pos1[i])
            new_pos2.append(pos2[i])
            new_slen.append(slen[i])
            new_y.append(y[i])

        if 1000 < lenth < 2000:
            c1 += 1
            new_word.append(word[i][:1000])
            new_word.append(word[i][1000:])

            new_pos1.append(pos1[i][:1000])
            new_pos1.append(pos1[i][1000:])

            new_pos2.append(pos2[i][:1000])
            new_pos2.append(pos2[i][1000:])

            new_slen.append(slen[i][:1000])
            new_slen.append(slen[i][1000:])

            new_y.append(y[i])
            new_y.append(y[i])

        if 2000 < lenth < 3000:
            c1 += 1
            new_word.append(word[i][:1000])
            new_word.append(word[i][1000:2000])
            new_word.append(word[i][2000:])

            new_pos1.append(pos1[i][:1000])
            new_pos1.append(pos1[i][1000:2000])
            new_pos1.append(pos1[i][2000:])

            new_pos2.append(pos2[i][:1000])
            new_pos2.append(pos2[i][1000:2000])
            new_pos2.append(pos2[i][2000:])

            new_slen.append(slen[i][:1000])
            new_slen.append(slen[i][1000:2000])
            new_slen.append(slen[i][2000:])

            new_y.append(y[i])
            new_y.append(y[i])
            new_y.append(y[i])

        if 3000 < lenth < 4000:
            c1 += 1
            new_word.append(word[i][:1000])
            new_word.append(word[i][1000:2000])
            new_word.append(word[i][2000:3000])
            new_word.append(word[i][3000:])

            new_pos1.append(pos1[i][:1000])
            new_pos1.append(pos1[i][1000:2000])
            new_pos1.append(pos1[i][2000:3000])
            new_pos1.append(pos1[i][3000:])

            new_pos2.append(pos2[i][:1000])
            new_pos2.append(pos2[i][1000:2000])
            new_pos2.append(pos2[i][2000:3000])
            new_pos2.append(pos2[i][3000:])

            new_slen.append(slen[i][:1000])
            new_slen.append(slen[i][1000:2000])
            new_slen.append(slen[i][2000:3000])
            new_slen.append(slen[i][3000:])

            new_y.append(y[i])
            new_y.append(y[i])
            new_y.append(y[i])
            new_y.append(y[i])

        if lenth > 4000:
            c1 += 1
            new_word.append(word[i][:1000])
            new_word.append(word[i][1000:2000])
            new_word.append(word[i][2000:3000])
            new_word.append(word[i][3000:4000])
            new_word.append(word[i][4000:])

            new_pos1.append(pos1[i][:1000])
            new_pos1.append(pos1[i][1000:2000])
            new_pos1.append(pos1[i][2000:3000])
            new_pos1.append(pos1[i][3000:4000])
            new_pos1.append(pos1[i][4000:])

            new_pos2.append(pos2[i][:1000])
            new_pos2.append(pos2[i][1000:2000])
            new_pos2.append(pos2[i][2000:3000])
            new_pos2.append(pos2[i][3000:4000])
            new_pos2.append(pos2[i][4000:])

            new_slen.append(slen[i][:1000])
            new_slen.append(slen[i][1000:2000])
            new_slen.append(slen[i][2000:3000])
            new_slen.append(slen[i][3000:4000])
            new_slen.append(slen[i][4000:])

            new_y.append(y[i])
            new_y.append(y[i])
            new_y.append(y[i])
            new_y.append(y[i])
            new_y.append(y[i])

    new_word = np.array(new_word)
    new_pos1 = np.array(new_pos1)
    new_pos2 = np.array(new_pos2)
    new_slen = np.array(new_slen)
    new_y = np.array(new_y)

    np.save('./data/m-ins/small_word.npy', new_word)
    np.save('./data/m-ins/small_pos1.npy', new_pos1)
    np.save('./data/m-ins/small_pos2.npy', new_pos2)
    np.save('./data/m-ins/small_len.npy', new_slen)
    np.save('./data/m-ins/small_y.npy', new_y)

    print '0-1000:{}, 1000-2000:{}, 2000-3000:{}, 3000-4000:{}, 4000<:{}'.format(
        c0, c1, c2, c3, c4
    )


# get answer metric for PR curve evaluation
def getans():
    test_y = np.load('./data/m-ins/testall_y.npy')
    eval_y = []
    for i in test_y:
        eval_y.append(i[1:])
    allans = np.reshape(eval_y, (-1))
    np.save('./data/m-ins/allans.npy', allans)


def get_metadata():
    with open('./origin_data/vectors/word_vec_50.pkl', 'rb') as f:
        word2vec = cPickle.load(f)
    fwrite = open('./data/metadata.tsv', 'wb')
    for k in sorted(word2vec.keys()):
        fwrite.write(k.encode('utf8') + '\n')
    fwrite.close()


if __name__ == '__main__':
    init()
    seperate()
    getsmall()
    getans()
    get_metadata()
