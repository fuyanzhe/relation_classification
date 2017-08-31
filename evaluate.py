# -*- encoding: utf-8 -*-
# Created by han on 17-8-11

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


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
    """
    get p, r, f1 by give prediction and answer, use_neg indicated whether p, r, f of NA relation is counted in Macro-avg
    """
    assert len(pred) == len(answer)
    confusion_matrix = get_confusion_matrix(pred, answer)
    p_r_f1_list, p_r_f1_macro, p_r_f1_micro = static_cm(confusion_matrix, use_neg)

    return p_r_f1_list, p_r_f1_macro, p_r_f1_micro


def get_wrong_ins(pred, ans, sen, p1, p2, x2id, id2x, id2rel, use_neg=True):
    """
    show wrong labeled instance
    """
    assert len(pred) == len(ans)
    pred = np.array(pred)
    answer = np.array([np.argmax(i) for i in ans])
    wrong_labeled_ins = np.asarray(sen)[pred != answer]
    wrong_labeled_lab = np.asarray(answer)[pred != answer]
    wrong_labeled_pre = np.asarray(pred)[pred != answer]

    max_sen_len = len(wrong_labeled_ins[0])
    # e1 pos
    wrong_labeled_pos1 = np.asarray(p1)[pred != answer]
    wrong_labeled_e1p = max_sen_len - wrong_labeled_pos1 + 1

    # e2 pos
    wrong_labeled_pos2 = np.asarray(p2)[pred != answer]
    wrong_labeled_e2p = max_sen_len - wrong_labeled_pos2 + 1

    wrong_ins = []
    wrong_ins_e1 = []
    wrong_ins_e2 = []
    wrong_ins_lab = []
    wrong_ins_pre = []

    blank_id = x2id['_BLANK']
    for idx in range(len(wrong_labeled_ins)):
        if (not use_neg) and wrong_labeled_lab[idx] == 0:
            continue
        else:
            ins = wrong_labeled_ins[idx][wrong_labeled_ins[idx] != blank_id]
            sen = []
            for id in ins:
                sen.append(id2x[id].encode('utf8'))
            wrong_ins.append(''.join(sen))
            wrong_ins_e1.append(id2x[wrong_labeled_ins[idx][wrong_labeled_e1p[idx]]].encode('utf8'))
            wrong_ins_e2.append(id2x[wrong_labeled_ins[idx][wrong_labeled_e2p[idx]]].encode('utf8'))
            wrong_ins_lab.append(id2rel[wrong_labeled_lab[idx]])
            wrong_ins_pre.append(id2rel[wrong_labeled_pre[idx]])

    wrong_labeled = zip(wrong_ins_lab, wrong_ins_pre, wrong_ins_e1, wrong_ins_e2, wrong_ins)

    return wrong_labeled


def save_prcurve(prob, answer, model_name, save_fn, use_neg=True):
    if not use_neg:
        prob_dn = []
        ans_dn = []
        for p in prob:
            prob_dn.append(p[1:])
        for ans in answer:
            ans_dn.append(ans[1:])
        prob = np.reshape(np.array(prob_dn), (-1))
        ans = np.reshape(np.array(ans_dn), (-1))
    else:
        prob = np.reshape(prob, (-1))
        ans = np.reshape(answer, (-1))

    precision, recall, threshold = precision_recall_curve(ans, prob)
    average_precision = average_precision_score(ans, prob)

    plt.clf()
    plt.plot(recall[:], precision[:], lw=2, color='navy', label=model_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.ylim([0.3, 1.0])
    # plt.xlim([0.0, 0.4])
    plt.title('Precision-Recall Area={0:0.2f}'.format(average_precision))
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(save_fn)
