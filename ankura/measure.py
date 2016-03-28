"""Functions for evaluating topics"""

from __future__ import division

import numpy
import scipy.sparse

class NaiveBayes(object):
    """A simple Multinomial Naive Bayes classifier"""

    def __init__(self, dataset, label_name):
        # Get label set
        labels = dataset.get_metadata(label_name)
        self.label_name = label_name
        self.labels = list(set(labels))
        label_indices = {l: i for i, l in enumerate(self.labels)}

        # Initialize counters
        label_counts = numpy.zeros(len(self.labels))
        word_counts = numpy.zeros((len(self.labels), dataset.vocab_size))

        # Compute label and word counts
        for doc, label in enumerate(labels):
            label_index = label_indices[label]
            label_counts[label_index] += 1
            tokens, _, counts = scipy.sparse.find(dataset.docwords[:, doc])
            for token, count in zip(tokens, counts):
                word_counts[label_index, token] += count

        # Normalize counts
        label_counts /= label_counts.sum()
        word_counts /= word_counts.sum(axis=1)[:, numpy.newaxis]

        # Store counts in log space, with 0's being mapped to -inf
        with numpy.errstate(divide='ignore'):
            self.label_counts = numpy.log(label_counts)
            self.word_counts = numpy.log(word_counts)

    def classify(self, data):
        """Returns the label with max posterior probability given the data"""
        posts = [self._log_posterior(l, data) for l in range(len(self.labels))]
        return self.labels[numpy.argmax(posts)]

    def _log_posterior(self, label_index, data):
        log_likelihood = 0
        tokens, _, counts = scipy.sparse.find(data)
        for token, count in zip(tokens, counts):
            log_likelihood += self.word_counts[label_index, token] * count
        return self.label_counts[label_index] + log_likelihood

    def validate(self, dataset, label_name=None):
        """Computes the accuracy of the classifier on the given data"""
        if not label_name:
            label_name = self.label_name
        labels = dataset.get_metadata(label_name)

        correct = 0
        for doc, label in enumerate(labels):
            predicted = self.classify(dataset.docwords[:, doc])
            if label == predicted:
                correct += 1
        return correct / len(labels)

    def contingency(self, dataset, label_name=None):
        """Constructs a ContingencyTable for the given data"""
        if not label_name:
            label_name = self.label_name
        gold_labels = dataset.get_metadata(label_name)

        data = []
        for doc, gold_label in enumerate(gold_labels):
            pred_label = self.classify(dataset.docwords[:, doc])
            data.append((gold_label, pred_label))

        return ContingencyTable(data, label_name)


def topic_coherence(word_indices, dataset, epsilon=0.01):
    """Measures the coherence of a single topic"""
    coherence = 0
    for i in word_indices:
        for j in word_indices:
            pair_count = dataset.docwords[i].multiply(dataset.docwords[j]).nnz
            count = dataset.docwords[j].nnz
            coherence += numpy.log((pair_count + epsilon) / count)
    return coherence


def n_choose_2(n):
    return (n * (n - 1)) / 2


def lim_plogp(p):
    if not p:
        return 0
    return p * numpy.log(p)


def lim_xlogy(x, y):
    if not x and not y:
        return 0
    return x * numpy.log(y)


class ContingencyTable(object):

    def __init__(self, labels, data):
        self.labels = list(labels)
        self.table = [[0 for _ in self.labels] for _ in self.labels]
        for gold, pred in data:
            self.table[gold][pred] += 1

    def fmeasure(self):
        gold_sums, pred_sums, total = self._sums()
        fmeasures = [0 for _ in self.labels]
        for gold_label in self.labels:
            for pred_label in self.labels:
                count = self.table[gold_label][pred_label]
                gold_count = gold_sums[gold_label]
                pred_count = pred_sums[pred_label]
                if count == 0 or gold_count == 0 or pred_count == 0:
                    continue
                recall = count / gold_count
                precision = count / pred_count
                fmeasure = recall * precision / (recall + precision)
                fmeasures[gold_label] = max(fmeasures[gold_label], fmeasure)

        result = 0
        for gold_label in self.labels:
            fmeasures[gold_label] * gold_sums[gold_label] / total
        return 2 * result

    def ari(self):
        gold_sum, pred_sum, ind_sum, all_sum = self._rand_sums()
        expected = gold_sum * pred_sum / all_sum
        maximum = (gold_sum+pred_sum) / 2
        return (ind_sum - expected) / (maximum - expected)

    def rand(self):
        gold_sum, pred_sum, ind_sum, all_sum = self._rand_sums()
        return (all_sum + 2 * ind_sum - gold_sum - pred_sum) / all_sum

    def vi(self):
        gold_sums, pred_sums, total = self._sums()

        gold_entropy = 0
        for count in gold_sums:
            gold_entropy -= lim_plogp(count / total)

        pred_entropy = 0
        for count in pred_sums:
            pred_entropy -= lim_plogp(count / total)

        mutual_info = 0
        for gold_label in self.labels:
            for pred_label in self.labels:
                count = self.table[gold_label][pred_label]
                joint_prob = count / total
                gold_prob = gold_sums[gold_label] / total
                pred_prob = pred_sums[pred_label] / total
                if gold_prob and pred_prob:
                    mutal = joint_prob / (gold_prob * pred_prob)
                    mutual_info += l

        return gold_entropy + pred_entropy - 2 * mutual_info

    def _sums(self):
        gold_sums = [0 for _ in self.labels]
        pred_sums = [0 for _ in self.labels]
        total = 0
        for gold_label in self.labels:
            for pred_label in self.labels:
                count = self.table[gold_label][pred_label]
                gold_sums[gold_label] += count
                pred_sums[pred_label] += count
                total += count
        return gold_sums, pred_sums, total

    def _rand_sums(self):
        gold_sums, pred_sums, total = self._sums()
        gold_sum = sum(n_choose_2(n) for n in gold_sums)
        pred_sum = sum(n_choose_2(n) for n in pred_sums)
        ind_sum = sum(n_choose_2(count) for row in self.table for count in row)
        all_sum = n_choose_2(total)
        return gold_sum, pred_sum, ind_sum, all_sum
