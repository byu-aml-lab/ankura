"""Functions for using and displaying topics"""

import numpy
import random
import collections


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


def _sample_categorical(counts):
    sample = random.uniform(0, sum(counts))
    for key, count in enumerate(counts):
        if sample < count:
            return key
        sample -= count
    raise ValueError(counts)


# POD types used for topic prediction
TokenTopic = collections.namedtuple('TokenTopic', 'token loc topic')
DocumentTheta = collections.namedtuple('DocumentTheta',
                                       'text tokens metadata theta')


def predict_topics(doc, topics, alpha=.01, num_iters=10):
    """Predicts token level topic assignments for a document

    Inference is performed using Gibbs sampling with Latent Dirichlet
    Allocation and fixed topics. A symetric Dirichlet prior over the
    document-topic distribution is used.
    """
    T = topics.shape[1]
    z = numpy.zeros(len(doc.tokens), dtype='uint')
    counts = numpy.zeros(T, dtype='uint')

    for n in range(len(doc.tokens)):
        z_n = random.randrange(T)
        z[n] = z_n
        counts[z_n] += 1

    def _prob(w_n, t):
        return (alpha + counts[t]) * topics[w_n, t]

    for _ in range(num_iters):
        for n, w_n in enumerate(doc.tokens):
            counts[z[n]] -= 1
            z[n] = _sample_categorical([_prob(w_n.token, t) for t in range(T)])
            counts[z[n]] += 1

    tokens = [TokenTopic(t.token, t.loc, z_n) for t, z_n in zip(doc.tokens, z)]
    theta = counts / counts.sum()
    return DocumentTheta(doc.text, tokens, doc.metadata, theta)


def topic_transform(corpus, topics, alpha=.01, num_iters=10):
    """Auguments a corpus so that it includes topic predictions"""
    corpus.documents[:] = [predict_topics(doc, topics, alpha, num_iters)
                           for doc in corpus.documents]
    return corpus
