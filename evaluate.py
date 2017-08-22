# -*- encoding: utf-8 -*-
# Created by han on 17-8-11

import numpy as np


def get_confusion_matrix(pred, answer):
    """
    get confusion_matrix
    """
    answer = list(answer)
    rel_num = len(answer[0])
    answer = [np.argmax(i) for i in answer]
    pred = list(pred)

    confusion_matrix = np.zeros([rel_num, rel_num])

    for res in zip(answer, pred):
        confusion_matrix[res[0], res[1]] += 1

    return confusion_matrix


def static_cm(confusion_matrix, use_neg=True):
    """
    get prf of each relation, calculate Macro-average prf and Micro-average prf
    """
    static_res = []
    rel_num = confusion_matrix.shape[0]
    tp_all, tp_fn_all, tp_fp_all = 0, 0, 0
    for i in range(rel_num):
        if (not use_neg) and i == 0:
            continue
        tp_fp = sum(confusion_matrix[:, i])
        tp_fn = sum(confusion_matrix[i, :])
        tp = confusion_matrix[i, i]
        tp_all += tp
        tp_fp_all += tp_fp
        tp_fn_all += tp_fn
        p = float(tp) / float(tp_fp) if tp_fp else 0
        r = float(tp) / float(tp_fn) if tp_fn else 0
        f1 = (2 * p * r) / (p + r) if p and r else 0
        if tp_fn or tp_fp:
            static_res.append([tp_fn, tp_fp, tp, p, r, f1, i])

    # Macro-average
    p_r_f1_macro = [0, 0, 0]
    contain_rel_num = len(static_res)
    for res in static_res:
        for x in (0, 1, 2):
            p_r_f1_macro[x] += res[x + 3]
    for x in (0, 1, 2):
        p_r_f1_macro[x] /= contain_rel_num

    # Micro-average
    p_all = tp_all / tp_fp_all
    r_all = tp_all / tp_fn_all
    f1_all = (2 * p_all * r_all) / (p_all + r_all)
    p_r_f1_micro = [p_all, r_all, f1_all]

    return static_res, p_r_f1_macro, p_r_f1_micro


def get_p_r_f1(pred, answer, use_neg=True):
    assert len(pred) == len(answer)
    confusion_matrix = get_confusion_matrix(pred, answer)
    p_r_f1_list, p_r_f1_macro, p_r_f1_micro = static_cm(confusion_matrix, use_neg)
    return p_r_f1_list, p_r_f1_macro, p_r_f1_micro


def get_wrong_ins(pred, test_data, word2id, id2word, id2rel, use_neg=True):
    assert len(pred) == len(test_data.y)
    pred = np.array(pred)
    answer = np.array([np.argmax(i) for i in test_data.y])
    wrong_labeled_ins = test_data.word[pred != answer]
    wrong_labeled_lab = answer[pred != answer]
    wrong_labeled_pre = pred[pred != answer]
    # e1 pos
    wrong_labeled_pos1 = test_data.pos1[pred != answer]
    wrong_labeled_pos1 = wrong_labeled_pos1[:, 0]
    wrong_labeled_e1p = 91 - wrong_labeled_pos1
    # e2 pos
    wrong_labeled_pos2 = test_data.pos2[pred != answer]
    wrong_labeled_pos2 = wrong_labeled_pos2[:, 0]
    wrong_labeled_e2p = 91 - wrong_labeled_pos2

    wrong_ins = []
    wrong_ins_e1 = []
    wrong_ins_e2 = []
    wrong_ins_lab = []
    wrong_ins_pre = []

    blank_id = word2id['_BLANK']
    for idx in range(len(wrong_labeled_ins)):
        if (not use_neg) and wrong_labeled_lab[idx] == 0:
            continue
        else:
            ins = wrong_labeled_ins[idx][wrong_labeled_ins[idx] != blank_id]
            sen = []
            for id in ins:
                sen.append(id2word[id].encode('utf8'))
            wrong_ins.append(''.join(sen))
            wrong_ins_e1.append(id2word[wrong_labeled_ins[idx][wrong_labeled_e1p[idx]]].encode('utf8'))
            wrong_ins_e2.append(id2word[wrong_labeled_ins[idx][wrong_labeled_e2p[idx]]].encode('utf8'))
            wrong_ins_lab.append(id2rel[wrong_labeled_lab[idx]])
            wrong_ins_pre.append(id2rel[wrong_labeled_pre[idx]])
    wrong_labeled = zip(wrong_ins_lab, wrong_ins_pre, wrong_ins_e1, wrong_ins_e2, wrong_ins)
    return wrong_labeled


def get_pr_curve(res_list):
    """
    res_list = [(predict_probability, relation)] * instance_number
    """
    tot = 0
    rel_num = len(res_list[0][1])
    prob_rel_list = list()
    pr_list = list()
    # exclude NA
    for i in range(len(res_list)):
        if res_list[i][1].index(1) != 0:
            tot += 1
        for j in range(1, rel_num):
            prob_rel_list.append((res_list[i][0][j], res_list[i][1][j]))
    sorted_list = sorted(prob_rel_list, reverse=True)
    for i in range(2000):
        correct = sum([j[1] for j in sorted_list[:(i + 1)]])
        pr_list.append((float(correct) / float(i + 1), float(correct) / float(tot)))
        if (i + 1) % 100 == 0:
            print "p: %f, r: %f" % (float(correct) / float(i + 1), float(correct) / float(tot))
    return pr_list


def get_pr_curve_bag(res_list):
    """
    res_list = [(entity_pair_idx, predict_probability, relation_set)] * bag_number
    or
    res_list = [(predict_probability, relation)] * bag_number
    """
    if len(res_list[0]) == 3:
        tot = 0
        rel_num = len(res_list[0][1])
        prob_rel_list = list()
        pr_list = list()
        # exclude NA
        for i in range(len(res_list)):
            if res_list[i][2] != {0}:
                tot += 1
            rel_vec = [0] * rel_num
            for rel_idx in res_list[i][2]:
                rel_vec[rel_idx] = 1
            for j in range(1, rel_num):
                prob_rel_list.append((res_list[i][1][j], rel_vec[j], j, res_list[i][0], res_list[i][2]))
        sorted_list = sorted(prob_rel_list, reverse=True)
        for i in range(min(5000, len(res_list))):
            correct = sum([j[1] for j in sorted_list[:(i + 1)]])
            pr_list.append(
                (float(correct) / float(i + 1), float(correct) / float(tot),
                 sorted_list[i][0], sorted_list[i][2], sorted_list[i][3], sorted_list[i][4])
            )
            if (i + 1) % 100 == 0:
                print "p: %f, r: %f" % (float(correct) / float(i + 1), float(correct) / float(tot))
        return pr_list
    else:
        tot = 0
        rel_num = len(res_list[0][0])
        prob_rel_list = list()
        pr_list = list()
        # exclude NA
        for i in range(len(res_list)):
            if res_list[i][1] != 0:
                tot += 1
            rel_vec = [0] * rel_num
            rel_vec[res_list[i][1]] = 1
            for j in range(1, rel_num):
                prob_rel_list.append((res_list[i][0][j], rel_vec[j], j, res_list[i][1]))
        sorted_list = sorted(prob_rel_list, reverse=True)
        for i in range(min(5000, len(res_list))):
            correct = sum([j[1] for j in sorted_list[:(i + 1)]])
            pr_list.append(
                (float(correct) / float(i + 1), float(correct) / float(tot),
                 sorted_list[i][0], sorted_list[i][2], sorted_list[i][3])
            )
            if (i + 1) % 100 == 0:
                print "p: %f, r: %f" % (float(correct) / float(i + 1), float(correct) / float(tot))
        return pr_list
