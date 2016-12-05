"""Functions for measuring topic quality"""

import numpy
import scipy.stats


def topic_coherence(word_indices, dataset, epsilon=.01):
    """Measures the coherence of a single topic"""
    coherence = 0
    for i in word_indices:
        for j in word_indices:
            pair_count = dataset.docwords[i].multiply(dataset.docwords[j]).nnz
            count = dataset.docwords[j].nnz
            coherence += numpy.log((pair_count + epsilon) / count)
    return coherence


def w_uniform(topic):
    """Measures the distance of the topic from the uniform distribution"""
    V = len(topic)
    return scipy.stats.entropy(topic, numpy.ones(V) / V)


def w_vacuous(topic, dataset):
    """Measures the distance of the topic from the vacuous distribution"""
    vacuous = dataset.Q.sum(axis=1) / dataset.Q.sum()
    return scipy.stats.entropy(topic, vacuous)


def d_bground(k, topic_transform):
    """Measures the distance of the topic from the background distribution"""
    D = topic_transform.num_docs
    bground = numpy.ones(D) / D

    topic = topic_transform.M[k, :]
    topic = numpy.array(topic.todense()).flatten()
    topic = topic.astype(float)
    topic /= topic.sum()

    return scipy.stats.entropy(topic, bground)
