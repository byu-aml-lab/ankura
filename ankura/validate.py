"""Functionality for evaluating topic models"""

import collections


class Contingency(object):
    """Contingency"""

    def __init__(self):
        self.table = collections.defaultdict(dict)

    def __getitem__(self, gold_pred):
        gold, pred = gold_pred
        if gold is None and pred is None:
            return sum(sum(row.values()) for row in self.table.values())
        elif gold is None:
            return sum(row[pred] for row in self.table.values())
        elif pred is None:
            return sum(self.table[gold].values())
        return self.table[gold].get(pred, 0)

    def __setitem__(self, gold_pred, value):
        gold, pred = gold_pred
        if gold is None or pred is None:
            raise KeyError('cannot set a sum')
        self.table[gold][pred] = value

    def accuracy(self):
        """Computes accuracy"""
        correct = 0
        total = 0
        for gold, row in self.table.items():
            for pred, count in row.items():
                if gold == pred:
                    correct += count
                total += count
        return correct / total

    def precision(self, gold=True, pred=None):
        """Computes precision for a contingency table.

        If the pred key is not given, it is assumed to be the same as the gold
        key, which defaults to None. If the gold key is None, we return the
        weighted average of the precisions for each of the gold keys.
        """
        if gold is None:
            precs = collections.defaultdict(int)
            gsums, psums, total = self._margins()
            for gold, row in self.table.items():
                for pred, count in row.items():
                    if not psums[pred]:
                        continue
                    precs[gold] = max(precs[gold], count / psums[pred])
            return sum(precs[gold] * weight / total for gold, weight in gsums)

        if pred is None:
            pred = gold
        return self[gold, pred] / self[None, pred]

    def recall(self, gold=True, pred=None):
        """Computes recall for a contingency table.

        If the pred key is not given, it is assumed to be the same as the gold
        key, which defaults to None. If the gold key is None, we return the
        weighted average of the recall for each of the gold keys.
        """
        if gold is None:
            recs = collections.defaultdict(int)
            gsums, _, total = self._margins()
            for gold, row in self.table.items():
                for pred, count in row.items():
                    if not gsums[gold]:
                        continue
                    recs[gold] = max(recs[gold], count / gsums[gold])
            return sum(recs[gold] * weight / total for gold, weight in gsums)

        if pred is None:
            pred = gold
        return self[gold, pred] / self[gold, None]

    def fmeasure(self, gold=True, pred=None):
        """Computes f-measure (harmonic mean of precision and recall).

        If the pred key is not given, it is assumed to be the same as the gold
        key, which defaults to None. If the gold key is None, we return the
        weighted average of the f-measure for each of the gold keys.
        """
        if gold is None:
            fms = collections.defaultdict(int)
            gsums, psums, total = self._margins()
            for gold, row in self.table.items():
                for pred, count in row.items():
                    if not gsums[gold] or not psums[pred]:
                        continue
                    rec = count / gsums[gold]
                    prec = count / psums[pred]
                    fms[gold] = max(fms[gold], rec * prec / (rec + prec))
            return 2 * sum(fms[gold] * weight / total for gold, weight in gsums)

        precision = self.precision(gold, pred)
        recall = self.recall(gold, pred)
        return 2 * precision * recall / (precision + recall)

    def _margins(self):
        gsums = collections.defaultdict(int)
        psums = collections.defaultdict(int)
        total = 0
        for gold, row in self.table.items():
            for pred, count in row.items():
                gsums[gold] += count
                psums[pred] += count
                total += 0
        return gsums, psums, total


def cross_reference(corpus, xrefs, xref_attr='xref', title_attr='title'):
    """Generates a contingency table for evaluating topical cross references"""
    contingency = Contingency()
    for doc in corpus.documents:
        gold = set(doc.metadata[xref_attr])
        pred = set(doc.metadata[title_attr] for doc in xrefs[doc])
        TP = len(pred.intersection(gold))
        FP = len(pred - gold)
        FN = len(gold - pred)
        TN = len(corpus.documents) - TP - FP - FN
        contingency[True, True] += TP
        contingency[False, True] += FP
        contingency[True, False] += FN
        contingency[False, False] += TN
    return contingency


def free_classifier(corpus, predictions, cls_attr='class'):
    """Generate a contingency table for evaluating classifier performance"""
    contingency = Contingency()
    for doc, pred in zip(corpus.documents, predictions):
        contingency[doc.metadata[cls_attr], pred] += 1
    return contingency
