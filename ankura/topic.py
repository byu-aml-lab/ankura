"""Functions for recovering anchor based topics from a coocurrence matrix"""

import scipy.sparse
import numpy
import random

from .pipeline import Dataset, filter_empty_words
from .util import sample_categorical


def logsum_exp(y):
    """Computes the sum of y in log space"""
    ymax = y.max()
    return ymax + numpy.log((numpy.exp(y - ymax)).sum())


_C1 = 1e-4
_C2 = .75

def exponentiated_gradient(Y, X, XX, epsilon):
    """Solves an exponentied gradient problem with L2 divergence"""
    XY = numpy.dot(X, Y)
    YY = float(numpy.dot(Y, Y))

    alpha = numpy.ones(X.shape[0]) / X.shape[0]
    old_alpha = numpy.copy(alpha)
    log_alpha = numpy.log(alpha)
    old_log_alpha = numpy.copy(log_alpha)

    AXX = numpy.dot(alpha, XX)
    AXY = float(numpy.dot(alpha, XY))
    AXXA = float(numpy.dot(AXX, alpha.transpose()))

    grad = 2 * (AXX - XY)
    old_grad = numpy.copy(grad)

    new_obj = AXXA - 2 * AXY + YY

    # Initialize book keeping
    stepsize = 1
    decreased = False
    convergence = float('inf')

    while convergence >= epsilon:
        old_obj = new_obj
        old_alpha = numpy.copy(alpha)
        old_log_alpha = numpy.copy(log_alpha)
        if new_obj == 0 or stepsize == 0:
            break

        # Add the gradient and renormalize in logspace, then exponentiate
        log_alpha -= stepsize * grad
        log_alpha -= logsum_exp(log_alpha)
        alpha = numpy.exp(log_alpha)

        # Precompute quantities needed for adaptive stepsize
        AXX = numpy.dot(alpha, XX)
        AXY = float(numpy.dot(alpha, XY))
        AXXA = float(numpy.dot(AXX, alpha.transpose()))

        # See if stepsize should decrease
        old_obj, new_obj = new_obj, AXXA - 2 * AXY + YY
        if new_obj > old_obj + _C1 * stepsize * numpy.dot(grad, alpha - old_alpha):
            stepsize /= 2.0
            alpha = old_alpha
            log_alpha = old_log_alpha
            new_obj = old_obj
            decreased = True
            continue

        # compute the new gradient
        old_grad, grad = grad, 2 * (AXX - XY)

        # See if stepsize should increase
        if numpy.dot(grad, alpha - old_alpha) < _C2 * numpy.dot(old_grad, alpha - old_alpha) and not decreased:
            stepsize *= 2.0
            alpha = old_alpha
            log_alpha = old_log_alpha
            grad = old_grad
            new_obj = old_obj
            continue

        # Update book keeping
        decreased = False
        convergence = numpy.dot(alpha, grad - grad.min())

    return alpha


def recover_topics(dataset, anchors, epsilon=2e-7):
    """Recovers topics given a cooccurence matrix and a set of anchor vectors"""
    # dont modify original Q
    Q = dataset.Q.copy()

    V = Q.shape[0]
    K = len(anchors)
    A = numpy.zeros((V, K))

    P_w = numpy.diag(Q.sum(axis=1))
    for word in range(V):
        if numpy.isnan(P_w[word, word]):
            P_w[word, word] = 1e-16

    # normalize the rows of Q to get Q_prime
    for word in range(V):
        Q[word, :] = Q[word, :] / Q[word, :].sum()

    # compute normalized anchors X, and precompute X * X.T
    X = anchors / anchors.sum(axis=1)[:, numpy.newaxis]
    XX = numpy.dot(X, X.transpose())

    for word in range(V):
        alpha = exponentiated_gradient(Q[word, :], X, XX, epsilon)
        if numpy.isnan(alpha).any():
            alpha = numpy.ones(K) / K
        A[word, :] = alpha

    # Use Bayes rule to compute topic matri
    # TODO(jeff) is this matrix conversion needed?
    A = numpy.matrix(P_w) * numpy.matrix(A)
    for k in range(K):
        A[:, k] = A[:, k] / A[:, k].sum()

    return numpy.array(A)


def predict_topics(topics, tokens, alpha=.01, rng=random, num_iters=10):
    """Produces topic assignments for a sequence tokens given a set of topics

    Inference is performed using Gibbs sampling. A uniform Dirichlet prior over
    the document topic distribution is used in the computation.
    """
    T = topics.shape[1]
    z = numpy.zeros(len(tokens), dtype='uint')
    counts = numpy.zeros(T)

    # init topics and topic counts
    for n in range(len(tokens)):
        z_n = rng.randrange(T)
        z[n] = z_n
        counts[z_n] += 1

    def _prob(w_n, t):
        return (alpha + counts[t]) * topics[w_n, t]

    for _ in range(num_iters):
        for n, w_n in enumerate(tokens):
            counts[z[n]] -= 1
            z[n] = sample_categorical([_prob(w_n, t) for t in range(T)])
            counts[z[n]] += 1

    return counts.astype('uint'), z


def topic_transform(topics, dataset, alpha=.01, rng=random):
    """Transforms a dataset to use topic assignments instead of tokens"""
    T = topics.shape[1]
    Z = numpy.zeros((T, dataset.num_docs), dtype='uint')
    for doc in range(dataset.num_docs):
        tokens = dataset.doc_tokens(doc)
        Z[:, doc], _ = predict_topics(topics, tokens, alpha, rng)
    Z = scipy.sparse.csc_matrix(Z)
    vocab = [str(i) for i in range(T)]
    return Dataset(Z, vocab, dataset.titles, dataset.metadata)


def topic_combine(topics, dataset, alpha=.01, rng=random):
    """Transforms a dataset to use token-topic features instead of tokens"""
    T = topics.shape[1]
    data = numpy.zeros((T*dataset.vocab_size, dataset.num_docs), dtype='uint')
    for doc in range(dataset.num_docs):
        tokens = dataset.doc_tokens(doc)
        _, assignments = predict_topics(topics, tokens, alpha, rng)
        for token, topic in zip(tokens, assignments):
            index = token * T + topic
            data[index, doc] += 1

    data = scipy.sparse.csc_matrix(data)
    vocab = ['{}-{}'.format(w, t) for w in dataset.vocab for t in range(T)]
    dataset = Dataset(data, vocab, dataset.titles, dataset.metadata)
    dataset = filter_empty_words(dataset)
    return dataset


def topic_summary_indices(topics, dataset, n=10):
    """Returns a list of the indices of the top n tokens per topic"""
    indices = []
    for k in range(topics.shape[1]):
        index = []
        for word in numpy.argsort(topics[:, k])[-n:][::-1]:
            index.append(word)
        indices.append(index)
    return indices


def topic_summary_tokens(topics, dataset, n=10):
    """Returns a list of top n tokens per topic"""
    summaries = []
    for index in topic_summary_indices(topics, dataset, n):
        summary = []
        for word in index:
            summary.append(dataset.vocab[word])
        summaries.append(summary)
    return summaries
