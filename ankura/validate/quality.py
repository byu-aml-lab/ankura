"""Functions for measuring topic quality"""

import numpy


def topic_coherence(word_indices, dataset, epsilon=.01):
    """Measures the coherence of a single topic"""
    coherence = 0
    for i in word_indices:
        for j in word_indices:
            pair_count = dataset.docwords[i].multiply(dataset.docwords[j]).nnz
            count = dataset.docwords[j].nnz
            coherence += numpy.log((pair_count + epsilon) / count)
    return coherence
