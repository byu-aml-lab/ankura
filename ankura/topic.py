"""Functions for using and displaying topics"""

import collections
import numpy


def topic_summary(topics, corpus=None, n=10):
    """Gets the top n tokens per topic.

    If a vocabulary is provided, the tokens are returned instead of the types.
    """
    summary = []
    for k in range(topics.shape[1]):
        index = []
        for word in numpy.argsort(topics[:, k])[-n:][::-1]:
            index.append(word)
        summary.append(index)

    if corpus:
        summary = [[corpus.vocabulary[w] for w in topic] for topic in summary]

    return summary


def predict_topics(corpus, topics, alpha=.01, num_iters=10):
    """Predicts token level topic assignments for a sequence of tokens

    Inference is performed using Gibbs sampling with Latent Dirichlet
    Allocation and fixed topics. A symetric Dirichlet prior over the
    document-topic distribution is used.
    """
