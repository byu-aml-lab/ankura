"""Functions for evaluating topics"""

from __future__ import division

import numpy
import scipy.sparse

class NaiveBayes(object):
    """A simple Multinomial Naive Bayes classifier"""

    def __init__(self, dataset, labels):
        # Get label set
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
        posts = [self._log_posterior(l, data) for l in xrange(len(self.labels))]
        return self.labels[numpy.argmax(posts)]

    def _log_posterior(self, label_index, data):
        log_likelihood = 0
        tokens, _, counts = scipy.sparse.find(data)
        for token, count in zip(tokens, counts):
            log_likelihood += self.word_counts[label_index, token] * count
        return self.label_counts[label_index] + log_likelihood

    def validate(self, dataset, labels):
        """Computes the accuracy of the classifier on the given data"""
        correct = 0
        for doc, label in enumerate(labels):
            predicted = self.classify(dataset.docwords[:, doc])
            if label == predicted:
                correct += 1
        return correct / len(labels)
