"""Implementation of the anchor algorithm by Arora et al. 2013"""

import collections
import numpy
import scipy.sparse
import scipy.stats


def build_cooccurrence(corpus):
    """Constructs a cooccurrence matrix from a Corpus"""
    # See supplementary 4.1 of Aurora et al. 2012 for more details
    V, D = len(corpus.vocabulary), len(corpus.documents)
    data, row, col = [], [], []
    H_hat = numpy.zeros(V)

    for i, doc in enumerate(corpus.documents):
        n_d = len(doc.tokens)
        if n_d <= 1:
            continue

        norm = 1 / (n_d * (n_d - 1))
        sqrt_norm = numpy.sqrt(norm)

        for t in doc.tokens:
            data.append(sqrt_norm)
            row.append(t.token)
            col.append(i)
            H_hat[t.token] += norm

    H_tilde = scipy.sparse.coo_matrix((data, (row, col)), (V, D)).tocsc()
    Q = H_tilde * H_tilde.T - numpy.diag(H_hat)
    return numpy.array(Q / D)


# pylint: disable=too-many-locals
def build_supervised_cooccurrence(corpus, attr_name, labeled_docs=None,
                                  label_weight=1, smoothing=1e-7):
    """Constructs a cooccurrence matrix from a labeled Corpus

    To find the label associated with a given document, this function looks at
    the document's metadata attribute for the value associated with the
    attr_name key. Thus, this function is for classification tasks, not
    regression.

    When labeled_docs is None or emtpy, all documents with attr_name as a key
    in their metadata are considered labeled; for semisupervised learning,
    specify only the documents you want labeled as an iterable of integers.

    To specify how strong a cooccurrence effect a label has with the tokens in
    the document, set label_weight accordingly. label_weight must be greater
    than zero.

    In the case of unlabeled documents, equal likelihood is given to all known
    classes, equal to smoothing. smoothing must be greater than zero.

    Returns cooccurrence matrix and a mapping from the label to its index
    """
    assert label_weight > 0, 'label_weight must be greater than zero'
    assert smoothing > 0, 'smoothing must be greater than zero'
    V, D = len(corpus.vocabulary), len(corpus.documents)
    data, row, col = [], [], []

    to_label = set(labeled_docs) if labeled_docs \
        else set(range(len(corpus.documents)))
    # account only for labels found in labeled set [via set comprehension]
    labels_found = {corpus.documents[i].metadata[attr_name] for i in to_label}
    label_types = {v: i+V for i, v in enumerate(sorted(labels_found))}

    H_hat = numpy.zeros(V + len(label_types))

    for i, doc in enumerate(corpus.documents):
        n_d = len(doc.tokens)
        if i in to_label:
            n_d += label_weight
        if n_d <= 1:
            continue

        norm = 1 / (n_d * (n_d - 1))
        sqrt_norm = numpy.sqrt(norm)

        # intentionally adding duplicate (row, col) entries
        for token in doc.tokens:
            data.append(sqrt_norm)
            row.append(token.token)
            col.append(i)
            H_hat[token.token] += norm

        if i in to_label:
            cur_type = label_types[corpus.documents[i].metadata[attr_name]]
            data.append(sqrt_norm * label_weight)
            row.append(cur_type)
            col.append(i)
            H_hat[cur_type] += norm * label_weight
        else:
            for cur_type in label_types.values():
                data.append(sqrt_norm * smoothing)
                row.append(cur_type)
                col.append(i)
                H_hat[cur_type] += norm * smoothing

    # duplicate entries are summed together when COO is converted to CSC
    H_tilde = scipy.sparse.coo_matrix((data, (row, col)),
                                      (V + len(label_types), D)).tocsc()
    Q = H_tilde * H_tilde.T - numpy.diag(H_hat)
    return numpy.array(Q / D), label_types


def _random_projection(A, k):
    R = numpy.random.choice([-1, 0, 0, 0, 0, 1], (A.shape[1], k))
    return numpy.dot(A, R * numpy.sqrt(3))


# pylint: disable=too-many-locals
def gram_schmidt(corpus, Q, k, doc_threshold=500, project_dim=1000, **kwargs):
    """Uses stabalized Gram-Schmidt decomposition to find k anchors."""
    # Find candidate anchors
    counts = collections.Counter()
    for doc in corpus.documents:
        counts.update(set(t.token for t in doc.tokens))
    candidates = [tid for tid, count in counts.items() if count > doc_threshold]

    # Row-normalize and project Q, preserving the original Q
    Q_orig = Q
    Q = Q / Q.sum(axis=1, keepdims=True)
    if project_dim:
        Q = _random_projection(Q, project_dim)

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
    if kwargs.get('return_indices'):
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


def _logsum_exp(y):
    ymax = y.max()
    return ymax + numpy.log((numpy.exp(y - ymax)).sum())


_C1 = 1e-4
_C2 = .75

def _exponentiated_gradient(Y, X, XX, epsilon):
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
        log_alpha -= _logsum_exp(log_alpha)
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


def recover_topics(Q, anchors, epsilon=2e-7):
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
