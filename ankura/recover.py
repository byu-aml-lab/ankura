"""perform recover"""
import numpy

def logsum_exp(y):
    """Computes the sum of y in log space"""
    m = y.max()
    return m + numpy.log((numpy.exp(y - m)).sum())


def exponentiated_gradient(Y, X, epsilon):
    """Solves an exponentied gradient problem with L2 divergence"""
    c1 = 10**(-4)
    c2 = 0.75

    XX = XX = numpy.dot(X, X.transpose())
    XY = numpy.dot(X, Y)
    YY = float(numpy.dot(Y, Y))

    K = X.shape[0]
    alpha = numpy.ones(K) / K

    old_alpha = numpy.copy(alpha)
    log_alpha = numpy.log(alpha)
    old_log_alpha = numpy.copy(log_alpha)

    AXX = numpy.dot(alpha, XX)
    AXY = float(numpy.dot(alpha, XY))
    AXXA = float(numpy.dot(AXX, alpha.transpose()))

    grad = 2 * (AXX - XY)
    new_obj = AXXA - 2 * AXY + YY

    old_grad = numpy.copy(grad)

    stepsize = 1
    decreased = False
    gap = float('inf')
    while gap >= epsilon:
        eta = stepsize
        old_obj = new_obj
        old_alpha = numpy.copy(alpha)
        old_log_alpha = numpy.copy(log_alpha)
        if new_obj == 0:
            break
        if stepsize == 0:
            break

        #update
        log_alpha -= eta*grad
        #normalize
        log_alpha -= logsum_exp(log_alpha)
        #compute new objective
        alpha = numpy.exp(log_alpha)

        AXX = numpy.dot(alpha, XX)
        AXY = float(numpy.dot(alpha, XY))
        AXXA = float(numpy.dot(AXX, alpha.transpose()))

        old_obj = new_obj
        new_obj = AXXA - 2 * AXY + YY
        if not new_obj <= old_obj + c1 * stepsize * numpy.dot(grad, alpha - old_alpha): #sufficient decrease
            stepsize /= 2.0
            alpha = old_alpha
            log_alpha = old_log_alpha
            new_obj = old_obj
            decreased = True
            continue

        #compute the new gradient
        old_grad = numpy.copy(grad)
        grad = 2*(AXX-XY)

        if (not numpy.dot(grad, alpha - old_alpha) >= c2 * numpy.dot(old_grad, alpha-old_alpha)) and (not decreased): #curvature
            stepsize *= 2.0 #increase stepsize
            alpha = old_alpha
            log_alpha = old_log_alpha
            grad = old_grad
            new_obj = old_obj
            continue

        decreased = False

        lam = numpy.copy(grad)
        lam -= lam.min()

        gap = numpy.dot(alpha, lam)

    return alpha


def recover_topics(Q, anchors, epsilon=1e-7):
    """recover"""
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

    for word in xrange(V):
        if word in anchors:
            alpha = numpy.zeros(K)
            alpha[anchors.index(word)] = 1
        else:
            alpha = exponentiated_gradient(Q[word, :], X, epsilon)
            if numpy.isnan(alpha).any():
                alpha = numpy.ones(K) / K
        A[word, :] = alpha

    A = numpy.matrix(P_w) * numpy.matrix(A)
    for k in xrange(K):
        A[:, k] = A[:, k]/A[:, k].sum()

    return numpy.array(A)
