"""Functions for using and displaying topics"""

import collections
import functools
import random
import sys

import numpy
import scipy.spatial

import ankura.util


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

    if not doc.tokens:
        return DocumentTheta(doc.text, [], doc.metadata, numpy.ones(T) / T)

    z = numpy.zeros(len(doc.tokens), dtype='uint')
    counts = numpy.zeros(T, dtype='uint')

    for n in range(len(doc.tokens)):
        z_n = random.randrange(T)
        z[n] = z_n
        counts[z_n] += 1

    for _ in range(num_iters):
        for n, w_n in enumerate(doc.tokens):
            counts[z[n]] -= 1
            cond = [alpha + counts[t] * topics[w_n, t] for t in range(T)]
            z[n] = ankura.util.sample_categorical(cond)
            counts[z[n]] += 1

    tokens = [TokenTopic(t.token, t.loc, z_n) for t, z_n in zip(doc.tokens, z)]
    theta = counts / counts.sum()
    return DocumentTheta(doc.text, tuple(tokens), doc.metadata, tuple(theta))


def topic_transform(corpus, topics, alpha=.01, num_iters=10):
    """Auguments a corpus so that it includes topic predictions"""
    corpus.documents[:] = [predict_topics(doc, topics, alpha, num_iters)
                           for doc in corpus.documents]

# TODO Add in option topic_transform using lda-c or scikit-learn like classtm


def cross_reference(corpus, doc=None, n=sys.maxsize, threshold=1):
    """Finds the nearest documents by topic similarity.

    If a document is given, then a list of references is returned for that
    document. Otherwise, cross references for each document in a corpus are
    given in a dict keyed by the documents. Consequently, the documents must be
    hashable.

    The closest n documents will be retruned (default=sys.maxsize). Documents
    whose similarity is behond the threshold (default=1) will not be returned.
    A threshold of 1 indicates that no filtering should be done, while a 0
    indicates that only exact topical matches should be returned. Note that
    the corpus must use DocumentTheta from predict_topics (or topic_transform).
    """
    def _xrefs(doc):
        dists = numpy.array([scipy.spatial.distance.cosine(doc.theta, d.theta)
                             if doc is not d else float('nan')
                             for d in corpus.documents])
        return list(corpus.documents[i] for i in dists.argsort()[:n]
                    if dists[i] <= threshold)

    if doc:
        return _xrefs(doc)
    else:
        return {doc: _xrefs(doc) for doc in corpus.documents}


def free_classifier(topics, Q, labels):
    """ASDF"""
    A_f = topics[:, -len(labels):]
    Q_L = Q[:]
    @functools.wraps(free_classifier)
    def _classifier(doc):
        return labels[numpy.argmax(A_f.dot(doc.theta) + Q_L.dot(doc.H))]
    return _classifier
