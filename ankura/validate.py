"""Functionality for evaluating topic models"""

import collections
import itertools

import scipy.stats
import numpy as np

from sklearn.linear_model import LogisticRegression


class Contingency(object):
    """Contingency is a table which gives the multivariate frequency
    distribution across known (or gold) labels and predicted labels.
    """

    def __init__(self):
        self.table = collections.defaultdict(dict)

    @staticmethod
    def from_cross_reference(corpus, xrefs, xref_attr='xref', title_attr='title'):
        """Generates a contingency table for evaluating cross references"""
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

    @staticmethod
    def from_classifier(corpus, classifier, label_attr='label'):
        contingency = Contingency()
        for doc in corpus.documents:
            contingency[doc.metadata[label_attr], classifier(doc)] += 1
        return contingency

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
        """Computes accuracy for a contingency table."""
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

    def vi(self):
        gsums, psums, total = self._margins()

        gentropy = 0
        for count in gsums.values():
            gentropy -= _lim_plogp(count / total)

        pentropy = 0
        for count in psums.values():
            pentropy -= _lim_plogp(count / total)

        mutual_info = 0
        for gold, row in self.table.items():
            for pred, count in row.items():
                joint_prob = count / total
                gprob = gsums[gold] / total
                pprob = psums[pred] / total
                if gprob and pprob:
                    mutual = joint_prob / (gprob * pprob)
                    mutual_info += _lim_xlogy(joint_prob, mutual)

        return gentropy + pentropy - 2 * mutual_info

    def _margins(self):
        gsums = collections.defaultdict(int)
        psums = collections.defaultdict(int)
        total = 0
        for gold, row in self.table.items():
            for pred, count in row.items():
                gsums[gold] += count
                psums[pred] += count
                total += count
        return gsums, psums, total

def anchor_accuracy(Q, anchors, test_corpus, train_corpus, label_name):
    num_topics = len(anchors)

    train_target = [doc.metadata[label_name] for doc in train_corpus.documents]
    test_target = [doc.metadata[label_name] for doc in test_corpus.documents]

    from .anchor import recover_topics
    from .topic import sampling_assign

    topics = recover_topics(Q, anchors, 1e-5)

    attr = 'z'
    sampling_assign(test_corpus, topics, z_attr=attr)
    sampling_assign(train_corpus, topics, z_attr=attr)

    test_matrix = scipy.sparse.lil_matrix((len(test_corpus.documents), num_topics * len(test_corpus.vocabulary)))
    train_matrix = scipy.sparse.lil_matrix((len(train_corpus.documents), num_topics * len(train_corpus.vocabulary)))

    for i, doc in enumerate(train_corpus.documents):
        for j, t in enumerate(doc.tokens):
            train_matrix[i, t.token * num_topics + doc.metadata[attr][j]] += 1

    for i, doc in enumerate(test_corpus.documents):
        for j, t in enumerate(doc.tokens):
            test_matrix[i, t.token * num_topics + doc.metadata[attr][j]] += 1

    lr = LogisticRegression()
    lr.fit(train_matrix, train_target)
    return lr.score(test_matrix, test_target)

def coherence(reference_corpus, topic_summary, epsilon=1e-2):
    """Computes topic coherence following Mimno et al., 2011 using pairwise log
    conditional probability taken from a reference corpus.

    Note that this is not the same as the NPMI based coherence proposed by
    Lau et al., 2014. Mimno et al., proposed using using the topic-modelled
    datadata, but one can optionally use an external corpus (e.g., Wikipedia)
    as proposed by Lau et al.

    The topic summary should be an array with each row giving the token types
    of the top words of each topic.
    """

    word_set = {word for topic in topic_summary for word in topic}
    counts = collections.Counter()
    pair_counts = collections.Counter()
    for doc in reference_corpus.documents:
        doc_set = set(tl.token for tl in doc.tokens).intersection(word_set)
        counts.update(doc_set)
        pair_counts.update(itertools.product(doc_set, doc_set))

    scores = []
    for topic in topic_summary:
        score = 0
        for i in topic:
            for j in topic:
                pair_count = pair_counts[(i, j)]
                count = counts[j]
                if count: 
                    score += np.log((pair_count + epsilon) / count)
        scores.append(score)
    return np.array(scores)


def _lim_plogp(p):
    if not p:
        return 0
    return p * np.log(p)


def _lim_xlogy(x, y):
    if not x and not y:
        return 0
    return x * np.log(y)


# Proposed Metrics for Token Level Topic Assignment

def topic_switch_percent(corpus, attr='z'):
    switches = 0
    n = 0
    for doc in corpus.documents:
        z = doc.metadata[attr]
        for a, b in zip(z, z[1:]):
            if a != b:
                switches += 1
            n += 1
    return switches / n


def topic_switch_vi(corpus, attr='z'):
    dist = Contingency()
    for doc in corpus.documents:
        z = doc.metadata[attr]
        for a, b in zip(z, z[1:]):
            dist[a, b] += 1
    return dist.vi()


def topic_word_divergence(corpus, topics, attr='z'):
    entropy = 0
    for doc in corpus.documents:
        if not doc.tokens:
            continue
        p = np.mean(topics[:, doc.metadata[attr]], axis=1)
        q = np.zeros_like(p)
        for t in doc.tokens:
            q[t.token] += 1
        q /= np.sum(q)
        m = (p + q) / 2
        entropy += scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)
    return entropy / 2 / len(corpus.documents)
