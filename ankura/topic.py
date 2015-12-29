"""Functions for recovering anchor based topics from a coocurrence matrix"""

import scipy.sparse
import numpy
import random

from .anchor import anchor_vectors
from .pipeline import Dataset


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


def recover_topics(dataset, anchors, epsilon=1e-7):
    """Recovers topics given a cooccurence matrix and a set of anchor vectors"""
    # dont modify original Q
    Q = dataset.Q.copy()

    V = Q.shape[0]
    K = len(anchors)
    A = numpy.zeros((V, K))

    P_w = numpy.diag(Q.sum(axis=1))
    for word in xrange(V):
        if numpy.isnan(P_w[word, word]):
            P_w[word, word] = 1e-16

    # normalize the rows of Q to get Q_prime
    for word in xrange(V):
        Q[word, :] = Q[word, :] / Q[word, :].sum()

    # compute normalized anchors X, and precompute X * X.T
    anchors = anchor_vectors(dataset, anchors)
    X = anchors / anchors.sum(axis=1)[:, numpy.newaxis]
    XX = numpy.dot(X, X.transpose())

    for word in xrange(V):
        alpha = exponentiated_gradient(Q[word, :], X, XX, epsilon)
        if numpy.isnan(alpha).any():
            alpha = numpy.ones(K) / K
        A[word, :] = alpha

    # Use Bayes rule to compute topic matrix
    A = numpy.matrix(P_w) * numpy.matrix(A) # TODO is matrix conversion needed?
    for k in xrange(K):
        A[:, k] = A[:, k] / A[:, k].sum()

    return numpy.array(A)


def predict_topics(topics, tokens, alpha=.01, rng=random):
    """Produces topic assignments for a sequence tokens given a set of topics

    Inference is performed using iterated conditional modes. A uniform
    Dirichlet prior over the document topic distribution is used in the
    computation.
    """
    T = topics.shape[1]
    z = numpy.zeros(len(tokens))
    counts = numpy.zeros(T, dtype='uint8')

    # init topics and topic counts
    for n in xrange(len(tokens)):
        z_n = rng.randrange(T)
        z[n] = z_n
        counts[z_n] += 1

    def _prob(w_n, t):
        return (alpha + counts[t]) * topics[w_n, t]

    # iterate until no further changes
    converged = False
    while not converged:
        converged = True
        for n, w_n in enumerate(tokens):
            counts[z[n]] -= 1
            z_n = max(xrange(T), key=lambda t: _prob(w_n, t)) # pylint:disable=cell-var-from-loop
            if z_n != z[n]:
                z[n] = z_n
                converged = False
            counts[z_n] += 1

    return counts


def topic_transform(topics, dataset, alpha=.01, rng=random):
    """Transforms a dataset to use topic assignments instead of tokens"""
    T = topics.shape[1]
    Z = numpy.zeros((T, dataset.num_docs), dtype='uint8')
    for doc in xrange(dataset.num_docs):
        counts = predict_topics(topics, dataset.doc_tokens(doc), alpha, rng)
        Z[:, doc] = counts
    Z = scipy.sparse.csc_matrix(Z)
    return Dataset(Z, [str(i) for i in xrange(T)], dataset.titles)
