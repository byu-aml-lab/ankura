"""Functions for recovering anchor based topics from a coocurrence matrix"""

import numpy


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

    stepsize = 1
    decreased = False
    gap = float('inf')

    while gap >= epsilon:
        old_obj = new_obj
        old_alpha = numpy.copy(alpha)
        old_log_alpha = numpy.copy(log_alpha)
        if new_obj == 0:
            break
        if stepsize == 0:
            break

        # Add the gradient and renormalize in logspace, then exponentiate
        log_alpha -= stepsize * grad
        log_alpha -= logsum_exp(log_alpha)
        alpha = numpy.exp(log_alpha)

        AXX = numpy.dot(alpha, XX)
        AXY = float(numpy.dot(alpha, XY))
        AXXA = float(numpy.dot(AXX, alpha.transpose()))

        old_obj, new_obj = new_obj, AXXA - 2 * AXY + YY
        if not new_obj <= old_obj + _C1 * stepsize * numpy.dot(grad, alpha - old_alpha):
            stepsize /= 2.0
            alpha = old_alpha
            log_alpha = old_log_alpha
            new_obj = old_obj
            decreased = True
            continue

        #compute the new gradient
        old_grad, grad = grad, 2 * (AXX - XY)

        if numpy.dot(grad, alpha - old_alpha) < _C2 * numpy.dot(old_grad, alpha-old_alpha) and not decreased:
            stepsize *= 2.0
            alpha = old_alpha
            log_alpha = old_log_alpha
            grad = old_grad
            new_obj = old_obj
            continue

        decreased = False
        gap = numpy.dot(alpha, grad - grad.min())

    return alpha


def recover_topics(Q, anchors, epsilon=1e-7):
    """Recovers topics given a cooccurence matrix and a set of anchor vectors"""
    V = Q.shape[0]
    K = len(anchors)
    A = numpy.zeros((V, K))

    P_w = numpy.diag(numpy.dot(Q, numpy.ones(V)))
    for word in xrange(V):
        if numpy.isnan(P_w[word, word]):
            P_w[word, word] = 10**(-16)

    #normalize the rows of Q_prime
    for word in xrange(V):
        Q[word, :] = Q[word, :] / Q[word, :].sum()

    X = Q[anchors, :]
    XX = numpy.dot(X, X.transpose())

    for word in xrange(V):
        if word in anchors:
            alpha = numpy.zeros(K)
            alpha[anchors.index(word)] = 1
        else:
            alpha = exponentiated_gradient(Q[word, :], X, XX, epsilon)
            if numpy.isnan(alpha).any():
                alpha = numpy.ones(K) / K
        A[word, :] = alpha

    A = numpy.matrix(P_w) * numpy.matrix(A)
    for k in xrange(K):
        A[:, k] = A[:, k]/A[:, k].sum()

    return numpy.array(A)


def print_summary(A, vocab, num_words=10, prefix=None):
    """Prints a summary of topics"""
    for k in xrange(A.shape[1]):
        topwords = numpy.argsort(A[:, k])[-num_words:][::-1]
        if prefix:
            print prefix(k) + ':',
        for word in topwords:
            print vocab[word],
        print
