"""Functions for evaluating topics"""

from __future__ import division

import subprocess
import threading
import collections
import functools
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

        return ContingencyTable(data)


def topic_coherence(word_indices, dataset, epsilon=0.01):
    """Measures the coherence of a single topic"""
    coherence = 0
    for i in word_indices:
        for j in word_indices:
            pair_count = dataset.docwords[i].multiply(dataset.docwords[j]).nnz
            count = dataset.docwords[j].nnz
            coherence += numpy.log((pair_count + epsilon) / count)
    return coherence


class ContingencyTable(object):
    """Computes various external clustering metrics on a contingency table"""

    def __init__(self, data):
        self.table = collections.defaultdict(counter)
        self.gold_labels = set()
        self.pred_labels = set()
        for gold, pred in data:
            self.gold_labels.add(gold)
            self.pred_labels.add(pred)
            self.table[gold][pred] += 1

    def fmeasure(self):
        """Computes the harmonic mean of precision and recall"""
        gold_sums, pred_sums, total = self._sums()
        fmeasures = counter()
        for gold in self.gold_labels:
            for pred in self.pred_labels:
                count = self.table[gold][pred]
                gold_sum = gold_sums[gold]
                pred_sum = pred_sums[pred]
                if count == 0 or gold_sum == 0 or pred_sum == 0:
                    continue
                recall = count / gold_sum
                precision = count / pred_sum
                fmeasure = recall * precision / (recall + precision)
                fmeasures[gold] = max(fmeasures[gold], fmeasure)

        result = 0
        for gold, gold_sum in gold_sums.items():
            result += fmeasures[gold] * gold_sum / total
        return 2 * result

    def ari(self):
        """Computes the chance ajdusted version of the rand index"""
        gold_sum, pred_sum, ind_sum, all_sum = self._rand_sums()
        expected = gold_sum * pred_sum / all_sum
        maximum = (gold_sum+pred_sum) / 2
        return (ind_sum - expected) / (maximum - expected)

    def rand(self):
        """Computes the rand index, which is essentially pair-wise accuracy"""
        gold_sum, pred_sum, ind_sum, all_sum = self._rand_sums()
        return (all_sum + 2 * ind_sum - gold_sum - pred_sum) / all_sum

    def vi(self):
        """Computes variation of information"""
        gold_sums, pred_sums, total = self._sums()

        gold_entropy = 0
        for count in gold_sums.values():
            gold_entropy -= lim_plogp(count / total)

        pred_entropy = 0
        for count in pred_sums.values():
            pred_entropy -= lim_plogp(count / total)

        mutual_info = 0
        for gold in self.gold_labels:
            for pred in self.pred_labels:
                count = self.table[gold][pred]
                joint_prob = count / total
                gold_prob = gold_sums[gold] / total
                pred_prob = pred_sums[pred] / total
                if gold_prob and pred_prob:
                    mutal = joint_prob / (gold_prob * pred_prob)
                    mutual_info += lim_xlogy(joint_prob, mutal)

        return gold_entropy + pred_entropy - 2 * mutual_info

    def _sums(self):
        gold_sums = {gold: 0 for gold in self.gold_labels}
        pred_sums = {pred: 0 for pred in self.pred_labels}
        total = 0
        for gold in self.gold_labels:
            for pred in self.pred_labels:
                count = self.table[gold][pred]
                gold_sums[gold] += count
                pred_sums[pred] += count
                total += count
        return gold_sums, pred_sums, total

    def _rand_sums(self):
        gold_sums, pred_sums, total = self._sums()

        gold_sum = sum(n_choose_2(n) for n in gold_sums.values())
        pred_sum = sum(n_choose_2(n) for n in pred_sums.values())

        ind_sum = 0
        for row in self.table.values():
            for count in row.values():
                ind_sum += n_choose_2(count)

        all_sum = n_choose_2(total)

        return gold_sum, pred_sum, ind_sum, all_sum


def n_choose_2(n):
    """Computes the binomial coefficient with k=2."""
    return (n * (n - 1)) / 2


def lim_plogp(p):
    """Computes p log p if p != 0, otherwise returns 0."""
    if not p:
        return 0
    return p * numpy.log(p)


def lim_xlogy(x, y):
    """Computes x log y of both x != 0 and y != 0, otherwise returns 0."""
    if not x and not y:
        return 0
    return x * numpy.log(y)


counter = functools.partial(collections.defaultdict, int)


def vowpal_accuracy(train, test, train_label, test_label=None):
    if test_label is None:
        test_label = train_label
    labels = list(set(train.get_metadata(train_label)))
    cmd = ['vw',
           '--quiet',
           '--loss_function', 'hinge',
           '--oaa', str(len(labels)),
           '/dev/stdin',
           '-p', '/dev/stdout']
    vw = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def write():
        for doc in range(train.num_docs):
            label = labels.index(train.doc_metadata(doc, train_label)) + 1
            data = ' '.join(train.vocab[t] for t in train.doc_tokens(doc))
            vw.stdin.write('{} 1 | {}\n'.format(label, data).encode())
        for doc in range(test.num_docs):
            label = labels.index(test.doc_metadata(doc, test_label)) + 1
            data = ' '.join(test.vocab[t] for t in test.doc_tokens(doc))
            vw.stdin.write('{} 0 | {}\n'.format(label, data).encode())
        vw.stdin.close()
    writer = threading.Thread(target=write)
    writer.start()

    correct = [0.0]
    def read():
        for doc in range(train.num_docs):
            vw.stdout.readline()
        for doc in range(test.num_docs):
            predicted = int(float(vw.stdout.readline())) - 1
            actual = labels.index(test.doc_metadata(doc, test_label))
            if predicted == actual:
                correct[0] += 1
    reader = threading.Thread(target=read)
    reader.start()

    writer.join()
    reader.join()
    return correct[0] / test.num_docs


def vowpal_contingency(train, test, train_label, test_label=None):
    if test_label is None:
        test_label = train_label
    labels = list(set(train.get_metadata(train_label)))
    cmd = ['vw',
           '--quiet',
           '--loss_function', 'hinge',
           '--oaa', str(len(labels)),
           '/dev/stdin',
           '-p', '/dev/stdout']
    vw = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def write():
        for doc in range(train.num_docs):
            label = labels.index(train.doc_metadata(doc, train_label)) + 1
            data = ' '.join(train.vocab[t] for t in train.doc_tokens(doc))
            vw.stdin.write('{} 1 | {}\n'.format(label, data).encode())
        for doc in range(test.num_docs):
            label = labels.index(test.doc_metadata(doc, test_label)) + 1
            data = ' '.join(test.vocab[t] for t in test.doc_tokens(doc))
            vw.stdin.write('{} 0 | {}\n'.format(label, data).encode())
        vw.stdin.close()
    writer = threading.Thread(target=write)
    writer.start()

    data = []
    def read():
        for doc in range(train.num_docs):
            vw.stdout.readline()
        for doc in range(test.num_docs):
            predicted = int(float(vw.stdout.readline())) - 1
            actual = labels.index(test.doc_metadata(doc, test_label))
            data.append((actual, predicted))
    reader = threading.Thread(target=read)
    reader.start()

    writer.join()
    reader.join()
    return ContingencyTable(data)
