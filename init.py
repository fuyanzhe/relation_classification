# -*- encoding: utf-8 -*-
# Created by han on 17-6-18
import numpy as np
import pandas as pd
import cPickle
from datetime import datetime


def get_word_emb():
    """
    get word embedding matrix and dict of word to embedding index
    :return: dict of word to embedding index
    """
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

    dim = len(vec[0]) if vec else 50
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec = np.array(vec, dtype=np.float32)
    np.save('./data/word_vec.npy', vec)

    with open('./data/word2id.pkl', 'wb') as f:
        cPickle.dump(word2id, f)

    id2word = {}
    for word, idx in word2id.iteritems():
        id2word[idx] = word
    with open('./data/id2word.pkl', 'wb') as f:
        cPickle.dump(id2word, f)

    return word2id


def get_char_emb():
    """
    get character embedding matrix and dict of character to embedding index
    :return: dict of character to embedding index
    """
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

    c_dim = 50
    c_vec.append(np.random.normal(size=c_dim, loc=0, scale=0.05))
    c_vec.append(np.random.normal(size=c_dim, loc=0, scale=0.05))
    c_vec = np.asarray(c_vec, dtype=np.float)
    # save vec matrix
    np.save('./data/char_vec.npy', c_vec)

    with open('./data/char2id.pkl', 'wb') as f:
        cPickle.dump(char2id, f)

    id2char = {}
    for char, idx in char2id.iteritems():
        id2char[idx] = char
    with open('./data/id2char.pkl', 'wb') as f:
        cPickle.dump(id2char, f)

    return char2id


def get_rel_idx():
    """
    get dict of relation to index
    :return: dict of relation to index
    """
    print 'reading relation to id...'
    with open('./origin_data/rel2idx.pkl', 'rb') as f:
        relation2id = cPickle.load(f)
    return relation2id


def get_metadata():
    """
    get words and characters in tsv which is used in tensorboard
    """
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


def pos_embed(x, max_len):
    """
    change position x to relative position in the position embedding matrix
    :param x: original position
    :param max_len: max relative distance
    :return: relative position
    """
    if x < -max_len:
        return 0
    if -max_len <= x <= max_len:
        return x + max_len + 1
    if x > max_len:
        return 2 * max_len + 2


def feature_word2char(sentence, en1pos, en2pos):
    """
    change word level sentence feature to character level sentence feature
    :param sentence: sentence by word
    :param en1pos: entity 1 position
    :param en2pos: entity 2 position
    :return: sentence by character, (entity 1 position, entity 2 position), sentence length in character level
    """
    sentence_c = []
    en1pos_c, en2pos_c = 0, 0
    sen_len_c = 0
    for i in range(len(sentence)):
        if i == en1pos:
            if '_con_' in sentence[i]:
                word_c = ''.join(sentence[i].split('_con_')).decode('utf8')
            else:
                word_c = sentence[i].decode('utf8')
            sentence_c.append(word_c)
            en1pos_c = sen_len_c
            sen_len_c += 1

        elif i == en2pos:
            if '_con_' in sentence[i]:
                word_c = ''.join(sentence[i].split('_con_')).decode('utf8')
            else:
                word_c = sentence[i].decode('utf8')
            sentence_c.append(word_c)
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

    return sentence_c, (en1pos_c, en2pos_c), sen_len_c


def sentence2idx(sentence, idx_dict):
    """
    translate word(characters) in the sentence to their index
    :param sentence:  sentence input by list
    :param idx_dict:  dict of word(character) to index
    :return: sentence by index
    """
    idx_list = []
    for i in sentence:
        idx_list.append(idx_dict[i] if i in idx_dict else idx_dict[u'_UNK'])
    return idx_list


def get_data_features(data_file, word2id, char2id, rel2id):
    """
    restructure data from original data file, get useful features
    :return: restructured data
    """
    # length of sentence is 100
    fixlen_w = 100
    # max length of position embedding is 100 (-100~+100)
    maxlen_w = 100
    # length of sentence by character is 330
    fixlen_c = 330
    # max length of position embedding by character is 330 (-330~+330)
    maxlen_c = 330

    # organize data by instance
    sen_all, sen_pos1_all, sen_pos2_all, sen_len_all = [], [], [], []
    sen_all_c, sen_pos1_all_c, sen_pos2_all_c, sen_len_all_c = [], [], [], []
    label_all = []

    # organize data by entity pair
    # {entity pair1: [
    #                   [[label1-sentence 1 index],[label1-sentence 2 index]...],
    #                   [[label2-sentence 1 index],[label2-sentence 2 index]...],
    #                   ...
    #                ]
    #  entity pair2: [...]}
    sen = {}
    # {entity pair: [label1,label2,...]} the label is one-hot vector
    ans = {}

    print datetime.now(), 'processing {}...'.format(data_file)
    all_index = 0
    data_train = pd.HDFStore(data_file)
    ep_counter_all = 0
    for rel in sorted(int(i.replace('/', '')) for i in data_train.keys()):
        data_rel = data_train['/' + str(rel)]
        ep_counter_r, ins_counter_r = 0, 0
        ep_set_r = set()
        for _, ins in data_rel.iterrows():
            ins_counter_r += 1
            # get entity name
            en1 = ins['uri1']
            en2 = ins['uri2']
            relation = ins['rel']
            if rel != 0:
                assert rel == rel2id[relation]

            # put the same entity pair sentences into a dict
            tup = (en1, en2)
            if tup not in ep_set_r:
                ep_counter_r += 1
                ep_set_r.add(tup)
            # label one-hot
            label = [0] * len(rel2id)
            label[rel] = 1

            # single-instance
            label_all.append(label)

            # multi-instance
            if tup not in sen:
                ep_counter_all += 1
                sen[tup] = [[]]
                ans[tup] = []
                ans[tup].append(label)
                label_tag = 0
            else:
                try:
                    label_tag = ans[tup].index(label)
                except ValueError:
                    sen[tup].append([])
                    ans[tup].append(label)
                    label_tag = len(ans[tup]) - 1

            # get sentence and entity pos
            sentence = ins['st_seg']
            en1pos = ins['ent1_p2']
            en2pos = ins['ent2_p2']
            sen_len_w = len(sentence)

            sentence_c, (en1pos_c, en2pos_c), sen_len_c = feature_word2char(sentence, en1pos, en2pos)
            output, output_c = [], []

            # get relative position
            for i in range(fixlen_w):
                word = word2id[u'_BLANK']
                rel_e1 = pos_embed(i - en1pos, maxlen_w)
                rel_e2 = pos_embed(i - en2pos, maxlen_w)
                output.append([word, rel_e1, rel_e2])

            # translate the words in sentences to index
            for i in range(min(fixlen_w, len(sentence))):
                if sentence[i].decode('utf8') not in word2id:
                    word = word2id[u'_UNK']
                else:
                    word = word2id[sentence[i].decode('utf8')]
                output[i][0] = word

            # get relative position
            for i in range(fixlen_c):
                char = char2id[u'_BLANK']
                rel_e1_c = pos_embed(i - en1pos_c, maxlen_c)
                rel_e2_c = pos_embed(i - en2pos_c, maxlen_c)
                output_c.append([char, rel_e1_c, rel_e2_c])

            # translate the characters in sentences to index
            for i in range(min(fixlen_c, len(sentence_c))):
                if sentence_c[i] not in char2id:
                    char = char2id[u'_UNK']
                else:
                    char = char2id[sentence_c[i]]
                output_c[i][0] = char

            # by instance
            output = np.asarray(output).T
            sen_idx, sen_p1, sen_p2 = output[0], output[1], output[2]
            sen_all.append(sen_idx)
            sen_pos1_all.append(sen_p1)
            sen_pos2_all.append(sen_p2)
            sen_len_all.append(sen_len_w)

            output_c = np.asarray(output_c).T
            sen_idx, sen_p1, sen_p2 = output_c[0], output_c[1], output_c[2]
            sen_all_c.append(sen_idx)
            sen_pos1_all_c.append(sen_p1)
            sen_pos2_all_c.append(sen_p2)
            sen_len_all_c.append(sen_len_c)

            # by entity pair
            sen[tup][label_tag].append(all_index)

            # increase index by one
            all_index += 1

        print 'rel_{}, entity pair number: {}, instance number: {}'.format(rel, ep_counter_r, ins_counter_r)
    print 'entity pair number all: {}, instance number all: {}'.format(ep_counter_all, all_index)
    data_train.close()

    feature_ins = (sen_all, sen_len_all, sen_pos1_all, sen_pos2_all)
    feature_ins_c = (sen_all_c, sen_len_all_c, sen_pos1_all_c, sen_pos2_all_c)
    feature_ep = (sen, ans)

    return label_all, feature_ins, feature_ins_c, feature_ep


def organize_ep2np(data_ep, file_name):
    """
    transform data organized by entity pair to numpy form
    :param data_ep: data organized by entity pair, data_ep[0] is sentence index, data_ep[1] is label
    :param file_name: file to show information about entities and their labels
    :return: data and label in numpy form
    """
    print 'organizing data {}...'.format(file_name)
    x_ep, y_ep = data_ep
    x_np, y_np = [], []
    with open('./data/m-ins/{}.txt'.format(file_name), 'w') as f:
        temp = 0
        for i in x_ep:
            assert len(x_ep[i]) == len(y_ep[i])
            # label number
            lab_num = len(x_ep[i])
            for j in range(lab_num):
                x_np.append(x_ep[i][j])
                y_np.append(y_ep[i][j])
                f.write('{}\t{}\t{}\t{}\t{}\n'.format(temp, i[0], i[1], np.argmax(y_ep[i][j]), x_ep[i][j]))
                temp += 1

    x_np = np.asarray(x_np)
    y_np = np.asarray(y_np)

    return x_np, y_np


def init():
    """
    process original data
    """
    # get index of word, character and relation
    word2id = get_word_emb()
    char2id = get_char_emb()
    rel2id = get_rel_idx()

    # process train data
    train_data_file = './origin_data/instances_rel_train.h5'

    train_label_all, train_feature_ins, train_feature_ins_c, train_feature_ep = get_data_features(
        train_data_file, word2id, char2id, rel2id
    )

    # process test data
    test_data_file = './origin_data/instances_rel_test.h5'

    test_label_all, test_feature_ins, test_feature_ins_c, test_feature_ep = get_data_features(
        test_data_file, word2id, char2id, rel2id
    )

    # multi-instance to numpy form
    train_x, train_y = organize_ep2np(train_feature_ep, 'train_q&a')
    test_x, test_y = organize_ep2np(test_feature_ep, 'test_q&a')

    print 'saving s-ins...'
    np.save('./data/s-ins/train_y.npy', np.asarray(train_label_all))
    np.save('./data/s-ins/train_word.npy', np.asarray(train_feature_ins[0]))
    np.save('./data/s-ins/train_len.npy', np.asarray(train_feature_ins[1]))
    np.save('./data/s-ins/train_pos1.npy', np.asarray(train_feature_ins[2]))
    np.save('./data/s-ins/train_pos2.npy', np.asarray(train_feature_ins[3]))
    np.save('./data/s-ins/train_char.npy', np.asarray(train_feature_ins_c[0]))
    np.save('./data/s-ins/train_len_c.npy', np.asarray(train_feature_ins_c[1]))
    np.save('./data/s-ins/train_pos1_c.npy', np.asarray(train_feature_ins_c[2]))
    np.save('./data/s-ins/train_pos2_c.npy', np.asarray(train_feature_ins_c[3]))

    np.save('./data/s-ins/test_y.npy', np.asarray(test_label_all))
    np.save('./data/s-ins/test_word.npy', np.asarray(test_feature_ins[0]))
    np.save('./data/s-ins/test_len.npy', np.asarray(test_feature_ins[1]))
    np.save('./data/s-ins/test_pos1.npy', np.asarray(test_feature_ins[2]))
    np.save('./data/s-ins/test_pos2.npy', np.asarray(test_feature_ins[3]))
    np.save('./data/s-ins/test_char.npy', np.asarray(test_feature_ins_c[0]))
    np.save('./data/s-ins/test_len_c.npy', np.asarray(test_feature_ins_c[1]))
    np.save('./data/s-ins/test_pos1_c.npy', np.asarray(test_feature_ins_c[2]))
    np.save('./data/s-ins/test_pos2_c.npy', np.asarray(test_feature_ins_c[3]))

    print 'saving m-ins...'
    np.save('./data/m-ins/train_x.npy', train_x)
    np.save('./data/m-ins/train_y.npy', train_y)
    np.save('./data/m-ins/test_x.npy', test_x)
    np.save('./data/m-ins/test_y.npy', test_y)


if __name__ == '__main__':
    init()
