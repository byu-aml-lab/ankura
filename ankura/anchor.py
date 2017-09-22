"""Implementation of the anchor algorithm and various anchor extensions"""

import collections
import numpy
import scipy.sparse
import scipy.stats

import ankura.util

def anchor_algorithm(corpus, k, doc_threshold=500, project_dim=1000):
    """Implementation of the anchor algorithm by Arora et al. 2013"""
    Q = build_cooccurrence(corpus)
    anchors = gram_schmidt_anchors(corpus, Q, k, doc_threshold, project_dim)
    return recover_topics(Q, anchors)


def build_cooccurrence(corpus):
    """Constructs a cooccurrence matrix from a Corpus"""
    V = len(corpus.vocabulary)
    Q = numpy.zeros((V, V))

    D = 0
    for doc in corpus.documents:
        n_d = len(doc.tokens)
        if n_d <= 1:
            continue
        D += 1

        norm = 1 / (n_d * (n_d - 1))
        for i, w_i in enumerate(doc.tokens):
            for j, w_j in enumerate(doc.tokens):
                if i == j:
                    continue
                Q[w_i.token, w_j.token] += norm

    return Q / D


def build_labeled_cooccurrence(corpus, attr_name, labeled_docs,
                               label_weight=1, smoothing=1e-7):
    """Constructs a cooccurrence matrix from a Corpus"""
    V = len(corpus.vocabulary)

    label_set = set()
    for d, doc in enumerate(corpus.documents):
        if d in labeled_docs:
            label_set.add(doc.metadata[attr_name])
    label_set = {l: V + i for i, l in enumerate(label_set)}

    K = len(label_set)
    Q = numpy.zeros((V+K, V+K))

    D = 0
    for d, doc in enumerate(corpus.documents):
        n_d = len(doc.tokens)
        if n_d <= 1:
            continue
        D += 1

        if d in labeled_docs:
            norm = 1 / 1
            index = label_set[doc.metadata[attr_name]]
            for i, w_i in enumerate(doc.tokens):
                for j, w_j in enumerate(doc.tokens):
                    if i == j:
                        continue
                    Q[w_i.token, w_j.token] += norm
                Q[w_i.token, index] += label_weight * norm
                Q[index, w_i.token] += label_weight * norm
            Q[index, index] += label_weight * (label_weight - 1) * norm
        else:
            norm = 1 / (n_d * (n_d + 2 * K * smoothing - 1) + K * (K * smoothing - smoothing))
            for i, w_i in enumerate(doc.tokens):
                for j, w_j in enumerate(doc.tokens):
                    if i == j:
                        continue
                    Q[w_i.token, w_j.token] += norm
                for j in label_set.values():
                    Q[w_i.token, j] += norm * smoothing
                    Q[j, w_i.token] += norm * smoothing
            for i in label_set.values():
                for j in label_set.values():
                    if i == j:
                        continue
                    Q[i, j] += norm * smoothing ** 2

    return Q / D, sorted(label_set, key=label_set.get)


# TODO Add QuickQ
# TODO Add SupAnk


def gram_schmidt_anchors(corpus, Q, k, doc_threshold=500, project_dim=1000, **kwargs):
    """Uses stabalized Gram-Schmidt decomposition to find k anchors."""
    # Find candidate anchors
    counts = collections.Counter()
    for doc in corpus.documents:
        counts.update(set(t.token for t in doc.tokens))
    candidates = [tid for tid, count in counts.items() if count > doc_threshold]
    k = min(k, len(candidates))

    # Row-normalize and project Q, preserving the original Q
    Q_orig = Q
    Q = Q / Q.sum(axis=1, keepdims=True)
    if project_dim:
        Q = ankura.util.random_projection(Q, project_dim)

    # Setup book keeping
    indices = numpy.zeros(k, dtype=numpy.int)
    basis = numpy.zeros((k-1, Q.shape[1]))

    # Find the farthest point from the origin
    max_dist = 0
    for i in candidates:
        dist = numpy.linalg.norm(Q[i])
        if dist > max_dist:
            max_dist = dist
            indices[0] = i

    # Translate all points to the new origin
    for i in candidates:
        Q[i] = Q[i] - Q[indices[0]]

    # Find the farthest point from origin
    max_dist = 0
    for i in candidates:
        dist = numpy.linalg.norm(Q[i])
        if dist > max_dist:
            max_dist = dist
            indices[1] = i
    basis[0] = Q[indices[1]] / max_dist

    # Stabilized gram-schmidt to finds new anchor words to expand the subspace
    for j in range(1, k - 1):
        # Project all the points onto the basis and find the farthest point
        max_dist = 0
        for i in candidates:
            Q[i] = Q[i] - numpy.dot(Q[i], basis[j-1]) * basis[j - 1]
            dist = numpy.dot(Q[i], Q[i])
            if dist > max_dist:
                max_dist = dist
                indices[j + 1] = i
                basis[j] = Q[i] / numpy.sqrt(numpy.dot(Q[i], Q[i]))

    # If requested, just return the indicies instead of anchor vectors
    if kwargs.get('return_indicies'):
        return indices

    # Use the original Q to extract anchor vectors using the anchor indices
    return Q_orig[indices, :]


def tandem_anchors(anchors, Q, corpus=None, epsilon=1e-10):
    """Creates pseudoword anchors from user provided anchor facets"""
    if corpus:
        anchor_indices = []
        for anchor in anchors:
            anchor_index = []
            for word in anchor:
                try:
                    anchor_index.append(corpus.vocabulary.index(word))
                except ValueError:
                    pass
            anchor_indices.append(anchor_index)
        anchors = anchor_indices

    basis = numpy.zeros((len(anchors), Q.shape[1]))
    for i, anchor in enumerate(anchors):
        basis[i] = scipy.stats.hmean(Q[anchor, :] + epsilon, axis=0)
    return basis


def _exponentiated_gradient(Y, X, XX, epsilon):
    _C1 = 1e-4
    _C2 = .75

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
        log_alpha -= ankura.util.logsumexp(log_alpha)
        alpha = numpy.exp(log_alpha)

        # Precompute quantities needed for adaptive stepsize
        AXX = numpy.dot(alpha, XX)
        AXY = float(numpy.dot(alpha, XY))
        AXXA = float(numpy.dot(AXX, alpha.transpose()))

        # See if stepsize should decrease
        old_obj, new_obj = new_obj, AXXA - 2 * AXY + YY
        if new_obj >= (
                old_obj + _C1 * stepsize * numpy.dot(grad, alpha - old_alpha)):
            stepsize /= 2.0
            alpha = old_alpha
            log_alpha = old_log_alpha
            new_obj = old_obj
            decreased = True
            continue

        # compute the new gradient
        old_grad, grad = grad, 2 * (AXX - XY)

        # See if stepsize should increase
        if not decreased and numpy.dot(grad, alpha - old_alpha) < (
                _C2 * numpy.dot(old_grad, alpha - old_alpha)):
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


def recover_topics(Q, anchors, epsilon=2e-6):
    """Recovers topics given a cooccurence matrix and a set of anchor vectors"""
    # dont modify original Q
    Q = Q.copy()

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
        alpha = _exponentiated_gradient(Q[word, :], X, XX, epsilon)
        if numpy.isnan(alpha).any():
            alpha = numpy.ones(K) / K
        A[word, :] = alpha

    # Use Bayes rule to compute topic matri
    A = numpy.dot(P_w, A)
    for k in range(K):
        A[:, k] = A[:, k] / A[:, k].sum()

    return numpy.array(A)
