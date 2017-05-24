"""Functionality for evaluating topic models"""

import sys
import collections
import math

from . import topic


class Contingency(object):
    """Contingency"""

    def __init__(self):
        self.table = collections.defaultdict(dict)

    def __getitem__(self, gold_pred):
        gold, pred = gold_pred
        return self.table[gold].get(pred, 0)

    def __setitem__(self, gold_pred, value):
        gold, pred = gold_pred
        self.table[gold][pred] = value

    def rand(self):
        """Computes the rand index, which is essentially pair-wise accuracy"""
        gold_sum, pred_sum, ind_sum, all_sum = self._rand_sums()
        return (all_sum + 2 * ind_sum - gold_sum - pred_sum) / all_sum

    def ari(self):
        """Computes the adjusted rand index"""
        gold_sum, pred_sum, ind_sum, all_sum = self._rand_sums()
        expected = gold_sum * pred_sum / all_sum
        maximum = (gold_sum+pred_sum) / 2
        return (ind_sum - expected) / (maximum - expected)

    def vinfo(self):
        """Computes variation of information"""
        gold_sums, pred_sums, total = self._sums()

        gold_entropy = 0
        for count in gold_sums.values():
            gold_entropy -= _lim_plogp(count / total)

        pred_entropy = 0
        for count in pred_sums.values():
            pred_entropy -= _lim_plogp(count / total)

        mutual_info = 0
        for gold, row in self.table.items():
            for pred, count in row.items():
                count = self.table[gold][pred]
                joint_prob = count / total
                gold_prob = gold_sums[gold] / total
                pred_prob = pred_sums[pred] / total
                if gold_prob and pred_prob:
                    mutal = joint_prob / (gold_prob * pred_prob)
                    mutual_info += _lim_xlogy(joint_prob, mutal)

        return gold_entropy + pred_entropy - 2 * mutual_info

    def _sums(self):
        gold_sums = collections.defaultdict(int)
        pred_sums = collections.defaultdict(int)
        total = 0
        for gold, row in self.table.items():
            for pred, count in row.items():
                gold_sums[gold] += count
                pred_sums[pred] += count
                total += count
        return gold_sums, pred_sums, total

    def _rand_sums(self):
        gold_sums, pred_sums, total = self._sums()

        gold_sum = sum(_nc2(n) for n in gold_sums.values())
        pred_sum = sum(_nc2(n) for n in pred_sums.values())

        ind_sum = 0
        for row in self.table.values():
            for count in row.values():
                ind_sum += _nc2(count)

        all_sum = _nc2(total)

        return gold_sum, pred_sum, ind_sum, all_sum


def eval_cross_reference(corpus, attr='xref', n=sys.maxsize, threshold=1):
    """Asdf"""
    contingency = Contingency()
    for doc in corpus.documents:
        pred = set(topic.cross_reference(doc, corpus, n, threshold))
        gold = set(doc.metadata[attr])
        TP = len(pred.intersection(gold))
        FP = len(pred - gold)
        FN = len(gold - pred)
        TN = len(corpus.documents) - TP - FP - FN
        contingency[True, True] += TP
        contingency[False, True] += FP
        contingency[True, False] += FN
        contingency[False, False] += TN
    return contingency


def _nc2(n):
    """Computes n choose 2"""
    return (n * (n - 1)) / 2


def _lim_plogp(p):
    """Computes p log p if p != 0, otherwise returns 0"""
    if not p:
        return 0
    return p * math.log(p)


def _lim_xlogy(x, y):
    """Computes x log y if both x != 0 and y != 0, otherwise returns 0"""
    if not x and not y:
        return 0
    return x * math.log(y)
