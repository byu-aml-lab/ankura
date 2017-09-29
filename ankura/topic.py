"""Functions for using and displaying topics"""

import functools
import sys

import numpy
import scipy.spatial
import sklearn.decomposition

import ankura.pipeline
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

    return numpy.array(summary)


def token_topics(doc, topics, alpha=.01, num_iters=10):
    """Predicts token level topic assignments for a document.

    Inference is performed using Gibbs sampling with Latent Dirichlet
    AllocationAllocation and fixed topics. A symetrics Dirichlet prior over the
    document-topic distribution is used.
    """
    if not doc.tokens:
        return []

    T = topics.shape[1]
    z = numpy.random.randint(T, size=len(doc.tokens), dtype='uint')

    counts = numpy.zeros(T, dtype='uint')
    for z_n in z:
        counts[z_n] += 1

    for _ in range(num_iters):
        for n, w_n in enumerate(doc.tokens):
            counts[z[n]] -= 1
            cond = [alpha + counts[t] * topics[w_n.token, t] for t in range(T)]
            z[n] = ankura.util.sample_categorical(cond)
            counts[z[n]] += 1

    return z


def document_topics(corpus_or_docwords, topics):
    """Predicts document-topic distributions for each document in a corpus.

    The input data can either be a corpus or a docwords matrix. If a corpus is
    given, a docwords matrix is constructed from that corpus. The
    document-topic distributions are given as a DxK matrix, with each row
    giving the topic distribution for a document.
    """
    V, K = topics.shape
    try:
        docwords = ankura.pipeline.build_docwords(corpus_or_docwords, V)
    except AttributeError:
        docwords = corpus_or_docwords

    lda = sklearn.decomposition.LatentDirichletAllocation(K)
    lda.components_ = topics.T
    lda._init_latent_vars(V)
    return lda.transform(docwords)


# TODO This now expects non-existant data type
def cross_reference(corpus, doc=None, n=sys.maxsize, threshold=1):
    """Finds the nearest documents by topic similarity.

    If a document is given, then a list of references is returned for that
    document. Otherwise, cross references for each document in a corpus are
    given in a dict keyed by the documents. Consequently, the documents must be
    hashable.

    The closest n documents will be returned (default=sys.maxsize). Documents
    whose similarity is behond the threshold (default=1) will not be returned.
    A threshold of 1 indicates that no filtering should be done, while a 0
    indicates that only exact topical matches should be returned. Note that
    the corpus must use DocumentTheta (obtained with topic_transform).
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


# XXX This seems to be broken!
def free_classifier(topics, Q, labels, epsilon=1e-7):
    """There is no free lunch, but this classifier is free"""
    K = len(labels)
    V = Q.shape[0] - K

    # Smooth and column normalize class-topic weights
    A_f = topics[-K:] + epsilon
    A_f /= A_f.sum(axis=0)

    # class_given_word
    Q = Q / Q.sum(axis=1, keepdims=True) # row-normalize Q without original
    Q_L = Q[-K:, :V]

    @functools.wraps(free_classifier)
    def _classifier(doc, theta):
        H = numpy.zeros(V)
        for w_d in doc.tokens:
            H[w_d.token] += 1

        topic_score = A_f.dot(theta)
        topic_score /= topic_score.sum(axis=0)

        word_score = Q_L.dot(H)
        word_score /= word_score.sum(axis=0)

        return labels[numpy.argmax(topic_score + word_score)]
    return _classifier
