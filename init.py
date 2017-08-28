# -*- encoding: utf-8 -*-
# Created by han on 17-6-18
import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict


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
    # word embedding
    print 'reading word embedding data...'
    vec = []
    word2id = {}
    with open('./origin_data/vectors/word_vec_50.pkl', 'rb') as f:
        word2vec = cPickle.load(f)
    for k in sorted(word2vec.keys()):
        word2id[k] = len(word2id)
        vec.append(word2vec[k])

    word2id[u'_UNK'] = len(word2id)
    word2id[u'_BLANK'] = len(word2id)

    with open('./data/word2id.pkl', 'wb') as f:
        cPickle.dump(word2id, f)

    id2word = {}
    for word, idx in word2id.iteritems():
        id2word[idx] = word
    with open('./data/id2word.pkl', 'wb') as f:
        cPickle.dump(id2word, f)

    dim = 50
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec = np.array(vec, dtype=np.float32)
    # save vec matrix
    np.save('./data/word_vec.npy', vec)

    # character level embedding
    print 'reading character embedding data...'
    c_vec = []
    char2id = {}
    with open('./origin_data/vectors/char_vec_50.pkl', 'rb') as f:
        char2vec = cPickle.load(f)
    for k in sorted(char2vec.keys()):
        char2id[k] = len(char2id)
        c_vec.append(char2vec[k])

    char2id[u'_UNK'] = len(char2id)
    char2id[u'_BLANK'] = len(char2id)

    with open('./data/char2id.pkl', 'wb') as f:
        cPickle.dump(char2id, f)

    id2char = {}
    for char, idx in char2id.iteritems():
        id2char[idx] = char
    with open('./data/id2char.pkl', 'wb') as f:
        cPickle.dump(id2char, f)

    dim = 50
    c_vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    c_vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    c_vec = np.array(c_vec, dtype=np.float32)
    # save vec matrix
    np.save('./data/char_vec.npy', c_vec)

    print 'reading relation to id'
    with open('./origin_data/rel2idx.pkl', 'rb') as f:
        relation2id = cPickle.load(f)

    # length of sentence is 100
    fixlen_w = 100
    # max length of position embedding is 100 (-100~+100)
    maxlen_w = 100

    # length of sentence by character is 330
    fixlen_c = 330
    # max length of position embedding by character is 330 (-330~+330)
    maxlen_c = 330

    # train data
    # single-instance
    train_sen_all = []
    train_sen_pos1_all = []
    train_sen_pos2_all = []
    train_sen_len_all = []

    train_sen_all_char = []
    train_sen_pos1_all_char = []
    train_sen_pos2_all_char = []
    train_sen_len_all_char = []

    train_label_all = []

    # multi-instance
    # {entity pair: {word: [[[label1-sentence 1],[label1-sentence 2]...],[[label2-sentence 1],[label2-sentence 2]...],
    #                char: [[[label1-sentence 1],[label1-sentence 2]...],[[label2-sentence 1],[label2-sentence 2]...]}}
    train_sen = defaultdict(dict)
    # {entity pair: {word: [[label1-sentence 1 length, label1-sentence 2 length],...],
    #                char: [[label1-sentence 1 length, label1-sentence 2 length],...]}}
    train_sen_len = defaultdict(dict)
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
                train_sen[tup]['word'] = [[]]
                train_sen[tup]['char'] = [[]]
                train_sen_len[tup]['word'] = [[]]
                train_sen_len[tup]['char'] = [[]]
                train_ans[tup] = []
                train_ans[tup].append(label)
                label_tag = 0
            else:
                temp = find_index(label, train_ans[tup])
                if temp == -1:
                    train_ans[tup].append(label)
                    label_tag = len(train_ans[tup]) - 1
                    train_sen[tup]['word'].append([])
                    train_sen_len[tup]['word'].append([])
                    train_sen[tup]['char'].append([])
                    train_sen_len[tup]['char'].append([])
                else:
                    label_tag = temp

            # get sentence and entity pos
            sentence = ins['st_seg']
            en1pos = ins['ent1_p2']
            en2pos = ins['ent2_p2']
            sen_len_w = len(sentence)

            sentence_c = []
            sen_len_c, en1pos_c, en2pos_c = 0, 0, 0
            for i in range(len(sentence)):
                if i == en1pos:
                    sentence_c.append(sentence[i].decode('utf8'))
                    en1pos_c = sen_len_c
                    sen_len_c += 1
                elif i == en2pos:
                    sentence_c.append(sentence[i].decode('utf8'))
                    en2pos_c = sen_len_c
                    sen_len_c += 1
                else:
                    if '_con_' in sentence[i]:
                        word_c = ''.join(sentence[i].split('_con_')).decode('utf8')
                        for c in word_c:
                            sentence_c.append(c)
                        sen_len_c += len(''.join(sentence[i].split('_con_')).decode('utf8'))

                    else:
                        sen_len_c += len(sentence[i].decode('utf8'))
                        for c in sentence[i].decode('utf8'):
                            sentence_c.append(c)

            output, output_c = [], []
            # get relative position
            for i in range(fixlen_w):
                word = word2id[u'_BLANK']
                rel_e1 = pos_embed(i - en1pos, maxlen_w)
                rel_e2 = pos_embed(i - en2pos, maxlen_w)
                output.append([word, rel_e1, rel_e2])
            for i in range(fixlen_c):
                char = char2id[u'_BLANK']
                rel_e1_c = pos_embed(i - en1pos_c, maxlen_c)
                rel_e2_c = pos_embed(i - en2pos_c, maxlen_c)
                output_c.append([char, rel_e1_c, rel_e2_c])

            # translate the words in sentences to index
            for i in range(min(fixlen_w, len(sentence))):
                if sentence[i].decode('utf8') not in word2id:
                    word = word2id[u'_UNK']
                else:
                    word = word2id[sentence[i].decode('utf8')]
                output[i][0] = word

            # translate the characters in sentences to index
            for i in range(min(fixlen_c, len(sentence_c))):
                if sentence_c[i] not in char2id:
                    char = char2id[u'_UNK']
                else:
                    char = char2id[sentence_c[i]]
                output_c[i][0] = char

            # single-instance
            sen_idx = [i[0] for i in output]
            sen_p1 = [i[1] for i in output]
            sen_p2 = [i[2] for i in output]
            sen_idx_c = [i[0] for i in output_c]
            sen_p1_c = [i[1] for i in output_c]
            sen_p2_c = [i[2] for i in output_c]
            train_sen_all.append(sen_idx)
            train_sen_all_char.append(sen_idx_c)
            train_sen_pos1_all.append(sen_p1)
            train_sen_pos2_all.append(sen_p2)
            train_sen_pos1_all_char.append(sen_p1_c)
            train_sen_pos2_all_char.append(sen_p2_c)
            train_sen_len_all.append(sen_len_w)
            train_sen_len_all_char.append(sen_len_c)
            # multi-instance
            train_sen[tup]['word'][label_tag].append(output)
            train_sen_len[tup]['word'][label_tag].append(sen_len_w)
            train_sen[tup]['char'][label_tag].append(output_c)
            train_sen_len[tup]['char'][label_tag].append(sen_len_c)

    data_train.close()

    # test data
    # single-instance
    test_sen_all = []
    test_sen_pos1_all = []
    test_sen_pos2_all = []
    test_sen_len_all = []
    test_sen_all_char = []
    test_sen_pos1_all_char = []
    test_sen_pos2_all_char = []
    test_sen_len_all_char = []
    test_label_all = []

    # multi-instance
    # {entity pair:[[sentence 1],[sentence 2]...]}
    test_sen = defaultdict(dict)
    # {entity pair:[sentence 1 length, sentence 2 length],...}
    test_sen_len = defaultdict(dict)
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
                test_sen[tup]['word'] = []
                test_sen_len[tup]['word'] = []
                test_sen[tup]['char'] = []
                test_sen_len[tup]['char'] = []
                test_ans[tup] = label
            else:
                test_ans[tup][y_id] = 1

            # get sentence and entity pos
            sentence = ins['st_seg']
            sen_len_w = len(sentence)
            en1pos = ins['ent1_p2']
            en2pos = ins['ent2_p2']
            sentence_c = []
            en1pos_c, en2pos_c = 0, 0
            sen_len_c = 0
            for i in range(len(sentence)):
                if i == en1pos:
                    sentence_c.append(sentence[i].decode('utf8'))
                    en1pos_c = sen_len_c
                    sen_len_c += 1
                elif i == en2pos:
                    sentence_c.append(sentence[i].decode('utf8'))
                    en2pos_c = sen_len_c
                    sen_len_c += 1
                else:
                    if '_con_' in sentence[i]:
                        word_c = ''.join(sentence[i].split('_con_')).decode('utf8')
                        for c in word_c:
                            sentence_c.append(c)
                        sen_len_c += len(''.join(sentence[i].split('_con_')).decode('utf8'))

                    else:
                        sen_len_c += len(sentence[i].decode('utf8'))
                        for c in sentence[i].decode('utf8'):
                            sentence_c.append(c)

            output, output_c = [], []

            # get relative position
            for i in range(fixlen_w):
                word = word2id[u'_BLANK']
                rel_e1 = pos_embed(i - en1pos, maxlen_w)
                rel_e2 = pos_embed(i - en2pos, maxlen_w)
                output.append([word, rel_e1, rel_e2])
            for i in range(fixlen_c):
                char = char2id[u'_BLANK']
                rel_e1_c = pos_embed(i - en1pos_c, maxlen_c)
                rel_e2_c = pos_embed(i - en2pos_c, maxlen_c)
                output_c.append([char, rel_e1_c, rel_e2_c])

            # translate the words in sentences to index
            for i in range(min(fixlen_w, len(sentence))):
                if sentence[i].decode('utf8') not in word2id:
                    word = word2id[u'_UNK']
                else:
                    word = word2id[sentence[i].decode('utf8')]
                output[i][0] = word

            # translate the characters in sentences to index
            for i in range(min(fixlen_c, len(sentence_c))):
                if sentence_c[i] not in char2id:
                    char = char2id[u'_UNK']
                else:
                    char = char2id[sentence_c[i]]
                output_c[i][0] = char

            # single-instance
            sen_idx = [i[0] for i in output]
            sen_p1 = [i[1] for i in output]
            sen_p2 = [i[2] for i in output]
            sen_idx_c = [i[0] for i in output_c]
            sen_p1_c = [i[1] for i in output_c]
            sen_p2_c = [i[2] for i in output_c]
            test_sen_all.append(sen_idx)
            test_sen_all_char.append(sen_idx_c)
            test_sen_pos1_all.append(sen_p1)
            test_sen_pos1_all_char.append(sen_p1_c)
            test_sen_pos2_all.append(sen_p2)
            test_sen_pos2_all_char.append(sen_p2_c)
            test_sen_len_all.append(sen_len_w)
            test_sen_len_all_char.append(sen_len_c)

            # multi-instance
            test_sen[tup]['word'].append(output)
            test_sen[tup]['char'].append(output_c)
            test_sen_len[tup]['word'].append(sen_len_w)
            test_sen_len[tup]['char'].append(sen_len_c)

    data_test.close()

    print 'saving s-ins...'
    # single-instance
    np.save('./data/s-ins/train_word.npy', np.array(train_sen_all))
    np.save('./data/s-ins/train_pos1.npy', np.array(train_sen_pos1_all))
    np.save('./data/s-ins/train_pos2.npy', np.array(train_sen_pos2_all))
    np.save('./data/s-ins/train_len.npy', np.array(train_sen_len_all))
    np.save('./data/s-ins/train_char.npy', np.array(train_sen_all_char))
    np.save('./data/s-ins/train_pos1_c.npy', np.array(train_sen_pos1_all_char))
    np.save('./data/s-ins/train_pos2_c.npy', np.array(train_sen_pos2_all_char))
    np.save('./data/s-ins/train_len_c.npy', np.array(train_sen_len_all_char))
    np.save('./data/s-ins/train_y.npy', np.array(train_label_all))

    np.save('./data/s-ins/test_word.npy', np.array(test_sen_all))
    np.save('./data/s-ins/test_pos1.npy', np.array(test_sen_pos1_all))
    np.save('./data/s-ins/test_pos2.npy', np.array(test_sen_pos2_all))
    np.save('./data/s-ins/test_len.npy', np.array(test_sen_len_all))
    np.save('./data/s-ins/test_char.npy', np.array(test_sen_all_char))
    np.save('./data/s-ins/test_pos1_c.npy', np.array(test_sen_pos1_all_char))
    np.save('./data/s-ins/test_pos2_c.npy', np.array(test_sen_pos2_all_char))
    np.save('./data/s-ins/test_len_c.npy', np.array(test_sen_len_all_char))
    np.save('./data/s-ins/test_y.npy', np.array(test_label_all))

    # multi-instance
    train_x = []
    train_x_c = []
    train_x_len = []
    train_x_len_c = []
    train_y = []

    test_x = []
    test_x_c = []
    test_x_len = []
    test_x_len_c = []
    test_y = []

    print 'organizing train data'
    f = open('./data/m-ins/train_q&a.txt', 'w')
    temp = 0
    for i in train_sen:
        assert len(train_ans[i]) == len(train_sen[i])
        # label number
        lenth = len(train_ans[i]['word'])
        for j in range(lenth):
            train_x.append(train_sen[i]['word'][j])
            train_x_c.append(train_sen[i]['char'][j])
            train_x_len.append(train_sen_len[i]['word'][j])
            train_x_len_c.append(train_sen_len[i]['char'][j])
            train_y.append(train_ans[i][j])
            f.write(str(temp) + '\t' + i[0] + '\t' + i[1] + '\t' + str(np.argmax(train_ans[i][j])) + '\n')
            temp += 1
    f.close()

    print 'organizing test data'
    f = open('./data/m-ins/test_q&a.txt', 'w')
    temp = 0
    for i in test_sen:
        test_x.append(test_sen[i]['word'])
        test_x_c.append(test_sen[i]['char'])
        test_x_len.append(test_sen_len[i]['word'])
        test_x_len_c.append(test_sen_len[i]['char'])
        test_y.append(test_ans[i])
        tempstr = ''
        for j in range(len(test_ans[i])):
            if test_ans[i][j] != 0:
                tempstr = tempstr + str(j) + '\t'
        f.write(str(temp) + '\t' + i[0] + '\t' + i[1] + '\t' + str(tempstr) + '\n')
        temp += 1
    f.close()

    train_x = np.array(train_x)
    train_x_c = np.array(train_x_c)
    train_x_len = np.array(train_x_len)
    train_x_len_c = np.array(train_x_len_c)
    train_y = np.array(train_y)

    test_x = np.array(test_x)
    test_x_c = np.array(test_x_c)
    test_x_len = np.array(test_x_len)
    test_x_len_c = np.array(test_x_len_c)
    test_y = np.array(test_y)

    print 'saving m-ins...'
    np.save('./data/m-ins/train_x.npy', train_x)
    np.save('./data/m-ins/train_x_c.npy', train_x_c)
    np.save('./data/m-ins/train_len.npy', train_x_len)
    np.save('./data/m-ins/train_len_c.npy', train_x_len_c)
    np.save('./data/m-ins/train_y.npy', train_y)

    np.save('./data/m-ins/testall_x.npy', test_x)
    np.save('./data/m-ins/testall_x_c.npy', test_x_c)
    np.save('./data/m-ins/testall_len.npy', test_x_len)
    np.save('./data/m-ins/testall_len_c.npy', test_x_len_c)
    np.save('./data/m-ins/testall_y.npy', test_y)


def seperate():
    # train
    print 'reading training data'
    x_train = np.load('./data/m-ins/train_x.npy')
    x_train_c = np.load('./data/m-ins/train_x_c.npy')

    train_word = []
    train_pos1 = []
    train_pos2 = []
    train_char = []
    train_pos1_c = []
    train_pos2_c = []

    print 'seprating train data'
    for tup_bag in x_train:
        char = []
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
            char.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        train_word.append(char)
        train_pos1.append(pos1)
        train_pos2.append(pos2)

    for tup_bag in x_train_c:
        char = []
        pos1 = []
        pos2 = []
        for sent in tup_bag:
            temp_char = []
            temp_pos1 = []
            temp_pos2 = []
            for char_feature in sent:
                temp_char.append(char_feature[0])
                temp_pos1.append(char_feature[1])
                temp_pos2.append(char_feature[2])
            char.append(temp_char)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        train_char.append(char)
        train_pos1_c.append(pos1)
        train_pos2_c.append(pos2)

    train_word = np.array(train_word)
    train_pos1 = np.array(train_pos1)
    train_pos2 = np.array(train_pos2)
    train_char = np.array(train_char)
    train_pos1_c = np.array(train_pos1_c)
    train_pos2_c = np.array(train_pos2_c)
    np.save('./data/m-ins/train_word.npy', train_word)
    np.save('./data/m-ins/train_pos1.npy', train_pos1)
    np.save('./data/m-ins/train_pos2.npy', train_pos2)
    np.save('./data/m-ins/train_char.npy', train_char)
    np.save('./data/m-ins/train_pos1_c.npy', train_pos1_c)
    np.save('./data/m-ins/train_pos2_c.npy', train_pos2_c)

    # test
    print 'seperating test all data'
    x_test = np.load('./data/m-ins/testall_x.npy')
    x_test_c = np.load('./data/m-ins/testall_x_c.npy')

    test_word = []
    test_pos1 = []
    test_pos2 = []
    test_char = []
    test_pos1_c = []
    test_pos2_c = []

    for tup_bag in x_test:
        char = []
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
            char.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        test_word.append(char)
        test_pos1.append(pos1)
        test_pos2.append(pos2)

    for tup_bag in x_test_c:
        char = []
        pos1 = []
        pos2 = []
        for sent in tup_bag:
            temp_char = []
            temp_pos1 = []
            temp_pos2 = []
            for char_feature in sent:
                temp_char.append(char_feature[0])
                temp_pos1.append(char_feature[1])
                temp_pos2.append(char_feature[2])
            char.append(temp_char)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        test_char.append(char)
        test_pos1_c.append(pos1)
        test_pos2_c.append(pos2)

    test_word = np.array(test_word)
    test_pos1 = np.array(test_pos1)
    test_pos2 = np.array(test_pos2)
    test_char = np.array(test_char)
    test_pos1_c = np.array(test_pos1_c)
    test_pos2_c = np.array(test_pos2_c)

    np.save('./data/m-ins/testall_word.npy', test_word)
    np.save('./data/m-ins/testall_pos1.npy', test_pos1)
    np.save('./data/m-ins/testall_pos2.npy', test_pos2)
    np.save('./data/m-ins/testall_char.npy', test_char)
    np.save('./data/m-ins/testall_pos1_c.npy', test_pos1_c)
    np.save('./data/m-ins/testall_pos2_c.npy', test_pos2_c)


# get answer metric for PR curve evaluation
def getans():
    test_y = np.load('./data/m-ins/testall_y.npy')
    eval_y = []
    for i in test_y:
        eval_y.append(i[1:])
    allans = np.reshape(eval_y, (-1))
    np.save('./data/m-ins/allans.npy', allans)


def get_metadata():
    # word
    with open('./origin_data/vectors/word_vec_50.pkl', 'rb') as f:
        word2vec = cPickle.load(f)
    with open('./data/metadata.tsv', 'wb') as fwrite:
        for k in sorted(word2vec.keys()):
            fwrite.write(k.encode('utf8') + '\n')
    # character
    with open('./origin_data/vectors/char_vec_50.pkl', 'rb') as f:
        char2vec = cPickle.load(f)
    with open('./data/metadata_c.tsv', 'wb') as fwrite:
        for k in sorted(char2vec.keys()):
            fwrite.write(k.encode('utf8') + '\n')


if __name__ == '__main__':
    init()
    seperate()
    getans()
    get_metadata()
