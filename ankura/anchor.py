"""Implementation of the anchor algorithm and various anchor extensions.

This class of algorithms was first described by Arora et al. 2013, but has been
modified and extended by various other authors.  Generally speaking, the
process of recovering topics using anchor words is as follows:
    * Construct a cooccurrence matrix, possibly with build_cooccurrence
    * Choose anchor words, possibly with gram_schmidt_anchors
    * Call recover_topics using a cooccurrence matrix and anchors
Variants of the anchor algorithm generally work by changing how the cooccurrence
matrix is constructed and/or how the anchor words are chosen.
"""

import collections
import numpy as np
import scipy.stats
import multiprocessing.pool

from . import util


def anchor_algorithm(corpus, k, doc_threshold=500, project_dim=1000):
    """Implementation of the anchor algorithm by Arora et al. 2013.

    This call builds a cooccurrence matrix from the given corpus, extracts k
    anchor words using the Gram-Schmidt process, and then recovers the
    topic-word distributions. The doc_threshold (default: 500) is how many
    documents a word must appear in to be considered as a potential anchor
    word, and project_dim (default: 1000, can be None) is the number of
    dimensions the Gram-Schmidt process should be performed under.

    This function is mostly for convenience, but is also a general guide for
    how to recover topics with this module.
    """
    Q = build_cooccurrence(corpus)
    anchors = gram_schmidt_anchors(corpus, Q, k, doc_threshold, project_dim)
    return recover_topics(Q, anchors)


def build_cooccurrence(corpus):
    """Constructs a cooccurrence matrix from a Corpus.

    The cooccurrence matrix is constructed following Anandkumar et al., 2012 in
    such a way that the cooccurrences of each document are given equal weight.
    We use a formulation of this process that allows us to consider each
    document individually, so that we can handle an arbitrarily large corpora.

    This cooccurrence matrix encodes the joint probabilities of observing two
    words together. Since the matrix is a joint distribution, it sums to 1.
    Note that if we row normalize this matrix, we would instead have
    conditional probabilities of observing a  word given we have already
    observed another word.
    """
    V = len(corpus.vocabulary)
    Q = np.zeros((V, V))

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
                               label_weight=1, smoothing=1e-7, **kwargs):
    """Constructs a cooccurrence matrix from a Corpus with labels.

    In addition to a corpus, this method requires that a label attribute name
    be given along with a set of labeled documents. For each label value, an
    invented pseudoword is included. The label_weight determines how many such
    pseudowords are added to each labeled document. Unlabeled documents are
    given a smoothing term for each label.

    Note that this algorithm must pass over the data twice. The first pass is
    to determine the set of label values, and the second pass actually
    constructs the cooccurrence matrix. However, since each document is
    considered individually, we can handle arbitrarily large corpora.
    """
    V = len(corpus.vocabulary)

    label_set = set()
    for d, doc in enumerate(corpus.documents):
        if d in labeled_docs:
            label_set.add(doc.metadata[attr_name])
    label_set = {l: V + i for i, l in enumerate(label_set)}

    L = len(label_set)
    Q = np.zeros((V+L, V+L))

    D = 0
    for d, doc in enumerate(corpus.documents):
        n_d = len(doc.tokens)
        if n_d <= 1:
            continue
        D += 1

        if d in labeled_docs:
            norm = 1 / ((n_d + label_weight) * (n_d + label_weight - 1))
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
            norm = 1 / (n_d * (n_d - 1) + 2 * n_d * L * smoothing + L * (L - 1) * smoothing**2)
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
                    Q[i, j] += norm * smoothing**2
    if kwargs.get('get_d'):
        return Q / D, sorted(label_set, key=label_set.get), D

    return Q / D, sorted(label_set, key=label_set.get)

def quick_Q(Q, corpus, attr_name, labeled_docs, newly_labeled_docs, labels, D,
                               label_weight=1, smoothing=1e-7):

    V = len(corpus.vocabulary)
    label_set = {l: V + i for i, l in enumerate(labels)}
    L = len(label_set)

    # Undo the normalization of Q (before we change D)
    Q = Q.copy() * D

    H = np.zeros((V+L, V+L))
    for d in newly_labeled_docs:
        doc = corpus.documents[d]
        n_d = len(doc.tokens)
        if n_d <= 1:
            continue

        # Subtract the unlabeled effect of this document
        norm = 1 / (n_d * (n_d - 1) + 2 * n_d * L * smoothing + L * (L - 1) * smoothing**2)
        for i, w_i in enumerate(doc.tokens):
            for j, w_j in enumerate(doc.tokens):
                if i == j:
                    continue
                H[w_i.token, w_j.token] -= norm
            for j in label_set.values():
                H[w_i.token, j] -= norm * smoothing
                H[j, w_i.token] -= norm * smoothing
        for i in label_set.values():
            for j in label_set.values():
                if i == j:
                    continue
                H[i, j] -= norm * smoothing**2

        # Add the labeled effect of this document
        norm = 1 / ((n_d + label_weight) * (n_d + label_weight - 1))
        index = label_set[doc.metadata[attr_name]]
        for i, w_i in enumerate(doc.tokens):
            for j, w_j in enumerate(doc.tokens):
                if i == j:
                    continue
                H[w_i.token, w_j.token] += norm
            H[w_i.token, index] += label_weight * norm
            H[index, w_i.token] += label_weight * norm
        H[index, index] += label_weight * (label_weight - 1) * norm
    Q += H
    return Q/D



def build_supervised_cooccurrence(corpus, attr_name, labeled_docs):

    V = len(corpus.vocabulary)
    Q = build_cooccurrence(corpus)

    label_set = set()
    for d, doc in enumerate(corpus.documents):
        if d in labeled_docs:
            label_set.add(doc.metadata[attr_name])
    label_set = {l : i for i, l in enumerate(label_set)}
    L = len(label_set)

    S = np.zeros((V, L))
    for d, doc in enumerate(corpus.documents):
        if d in labeled_docs:
            label_index = label_set[doc.metadata[attr_name]]
            for i, w_i in enumerate(doc.tokens):
                S[w_i.token, label_index] += 1

    for i in range(S.shape[0]):

        row_sum = np.sum(S[i,:])
        if row_sum == 0:
            continue
        S[i,:] /= row_sum

    return np.hstack((Q, S))


def gram_schmidt_anchors(corpus, Q, k, doc_threshold=500, project_dim=1000, **kwargs):
    """Uses stabilized Gram-Schmidt decomposition to find k anchors.

    Each row of Q represents a word embedded in V-dimensional space, with each
    dimensions encoding a cooccurrence probability. We first pick the two most
    extreme points to serve as a new origin and basis. We then iteratively add
    new points to our basis by selecting the point which is furthest away after
    projecting it onto the current span.

    We do this until we have selected k anchor words. However, only words which
    occur in at least doc_threshold (default: 500) documents will be
    considered. For computational efficiency, we first project our points in to
    project_dim (default: 1000, optionally None) dimensional space.

    The rows of the cooccurrence matrix Q corresponding to the selected anchor
    words are returned. If the keyword argument 'return_indices' is True, we
    return the row indices instead of the rows themselves.
    """
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
        Q = util.random_projection(Q, project_dim)

    # Setup book keeping
    indices = np.zeros(k, dtype=np.int)
    basis = np.zeros((k-1, Q.shape[1]))

    # Find the farthest point from the origin
    max_dist = 0
    for i in candidates:
        dist = np.linalg.norm(Q[i])
        if dist > max_dist:
            max_dist = dist
            indices[0] = i

    # Translate all points to the new origin
    Q[candidates] -= Q[indices[0]]

    # Find the farthest point from origin
    max_dist = 0
    for i in candidates:
        dist = np.linalg.norm(Q[i])
        if dist > max_dist:
            max_dist = dist
            indices[1] = i
    basis[0] = Q[indices[1]] / max_dist

    # Stabilized gram-schmidt to finds new anchor words to expand the subspace
    for j in range(1, k - 1):
        # Project all the points onto the basis and find the farthest point
        max_dist = 0
        for i in candidates:
            Q[i] = Q[i] - np.dot(Q[i], basis[j-1]) * basis[j - 1]
            dist = np.dot(Q[i], Q[i])
            if dist > max_dist:
                max_dist = dist
                indices[j + 1] = i
        basis[j] = Q[indices[j + 1]] / np.sqrt(max_dist)

    # If requested, just return the indices instead of anchor vectors
    if kwargs.get('return_indices'):
        return indices

    # Use the original Q to extract anchor vectors using the anchor indices
    return Q_orig[indices, :]


def tandem_anchors(anchors, Q, corpus=None, epsilon=1e-10):
    """Creates pseudoword anchors from user provided anchor facets.

    The anchors should be a list of list of row indices. Each list of indices
    is a multiword anchor which is constructed by taking the harmonic mean of
    the indexed rows. To avoid zero weights, an epsilon (default: 1e-10) is
    added to each anchor vector.
    """
    # TODO document why the corpus is here...
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

    basis = np.zeros((len(anchors), Q.shape[1]))
    for i, anchor in enumerate(anchors):
        basis[i] = scipy.stats.hmean(Q[anchor, :] + epsilon, axis=0)
    return basis

@util.jit
def _exponentiated_gradient(Y, X, XX, epsilon):
    _C1 = 1e-4
    _C2 = .75

    XY = np.dot(X, Y)
    YY = np.dot(Y, Y)

    alpha = np.ones(X.shape[0]) / X.shape[0]
    old_alpha = np.copy(alpha)
    log_alpha = np.log(alpha)
    old_log_alpha = np.copy(log_alpha)

    AXX = np.dot(alpha, XX)
    AXY = np.dot(alpha, XY)
    AXXA = np.dot(AXX, alpha.transpose())

    grad = 2 * (AXX - XY)
    old_grad = np.copy(grad)

    new_obj = AXXA - 2 * AXY + YY

    # Initialize book keeping
    stepsize = 1
    decreased = False
    convergence = np.inf

    while convergence >= epsilon:
        old_obj = new_obj
        old_alpha = np.copy(alpha)
        old_log_alpha = np.copy(log_alpha)
        if new_obj == 0 or stepsize == 0:
            break

        # Add the gradient and renormalize in logspace, then exponentiate
        log_alpha -= stepsize * grad
        log_alpha -= util.logsumexp(log_alpha)
        alpha = np.exp(log_alpha)

        # Precompute quantities needed for adaptive stepsize
        AXX = np.dot(alpha, XX)
        AXY = np.dot(alpha, XY)
        AXXA = np.dot(AXX, alpha.transpose())

        # See if stepsize should decrease
        old_obj, new_obj = new_obj, AXXA - 2 * AXY + YY
        if new_obj >= (
                old_obj + _C1 * stepsize * np.dot(grad, alpha - old_alpha)):
            stepsize /= 2.0
            alpha = old_alpha
            log_alpha = old_log_alpha
            new_obj = old_obj
            decreased = True
            continue

        # compute the new gradient
        old_grad, grad = grad, 2 * (AXX - XY)

        # See if stepsize should increase
        if not decreased and np.dot(grad, alpha - old_alpha) < (
                _C2 * np.dot(old_grad, alpha - old_alpha)):
            stepsize *= 2.0
            alpha = old_alpha
            log_alpha = old_log_alpha
            grad = old_grad
            new_obj = old_obj
            continue

        # Update book keeping
        decreased = False
        convergence = np.dot(alpha, grad - grad.min())

    if np.isnan(alpha).any():
        alpha = np.ones(X.shape[0]) / X.shape[0]
    return alpha


def recover_topics(Q, anchors, epsilon=2e-6, **kwargs):
    """Recovers topics given a cooccurrence matrix and a set of anchor vectors.

    We represent each word (rows of the cooccurrence matrix Q) as a convex
    combination of the anchors. Each convex combination is computed using
    exponentiated gradient descent. These combination weights gives us the
    inverse conditioning we need, so we multiply the weights with the prior
    probabilities of each word to obtain a topic-word matrix which gives
    probabilities of word given topic.

    Since the computation of the convex combinations for each word is
    embarassingly parallel, it can be faster to use multiprocessing. The
    keyword argument parallelism can used to specify the number of threads
    used. The keyword argument chunksize can also be given to specify the
    approximate number of words sent to a thread to work on at a time.
    """
    # Don't modify original Q
    Q = Q.copy()

    # Get dimensions of topic matrix
    V = Q.shape[0]
    K = len(anchors)

    # Compute prior probability of each word with row sums of Q.
    P_w = np.diag(Q.sum(axis=1))
    for word in range(V):
        if np.isnan(P_w[word, word]):
            P_w[word, word] = 1e-16

    # Normalize the rows of Q to get Q_prime
    for word in range(V):
        Q[word, :] = Q[word, :] / Q[word, :].sum()

    # Compute normalized anchors X, and precompute X * X.T
    X = anchors / anchors.sum(axis=1)[:, np.newaxis]
    XX = np.dot(X, X.transpose())

    # Represent each word as a convex combination of anchors.
    parallelism = kwargs.get('parallelism')
    if parallelism:
        worker = lambda word: _exponentiated_gradient(Q[word], X, XX, epsilon)
        chunksize = kwargs.get('chunksize', V // parallelism)
        with multiprocessing.pool.ThreadPool(parallelism) as pool:
            C = pool.map(worker, range(V), chunksize)
        C = np.array(C)
    else:
        C = np.zeros((V, K))
        for word in range(V):
            C[word] = _exponentiated_gradient(Q[word], X, XX, epsilon)

    # Use Bayes rule to compute topic matrix
    A = np.dot(P_w, C)
    for k in range(K):
        A[:, k] = A[:, k] / A[:, k].sum()

    if kwargs.get('get_c'):
        return C.transpose(), A
    return A
