"""Runs anchor words and stuff"""

import numpy

def construct_Q(M):
    """Constructs a cooccurrence matrix from a sparse docword matrix

    The docword matrix M is expected to be a sparse scipy matrix which can be
    converted to csc form. The output matrix will be a 2d numpy array.
    """
    vocab_size, num_docs = M.shape

    # See supplementary 4.1 of Aurora et. al. 2012 for information on these
    tilde_H = M.tocsc()
    hat_H = numpy.zeros(vocab_size)

    # Construct tilde_H and hat_H
    for j in range(tilde_H.indptr.size - 1):
        # get indices of column j
        col_start = tilde_H.indptr[j]
        col_end = tilde_H.indptr[j + 1]
        row_indices = tilde_H.indices[col_start: col_end]

        # get count of tokens in column (document) and compute norm
        count = numpy.sum(tilde_H.data[col_start: col_end])
        norm = count * (count - 1)

        # update hat_H and tilde_H (see supplementary)
        hat_H[row_indices] = tilde_H.data[col_start: col_end] / norm
        tilde_H.data[col_start: col_end] /= numpy.sqrt(norm)


    # construct and return normalized Q
    Q = tilde_H * tilde_H.transpose() - numpy.diag(hat_H)
    return numpy.array(Q / num_docs)


def random_projection(A, k, rng=numpy.random):
    """Randomly reduces the dimensionality of a n x d matrix A to k x d

    We follow the method given by Achlioptas 2001 which yields a projection
    which does well at preserving pairwise distances within some small factor.
    We do this by multiplying A with R, a n x k matrix with each element
    R_{i,j} distributed as:
        sqrt(3)  with probability 1/6
        0        with probability 2/3
        -sqrt(3) with probability 1/ 6
    The resulting matrix therefore has the dimensions k x d so each of the d
    examples in A is reduced from n dimensions to k dimensions.
    """
    R = rng.choice([-1, 0, 0, 0, 0, 1], (A.shape[1], k)) * numpy.sqrt(3)
    return numpy.dot(A, R)


def identify_candidates(M, doc_threshold):
    """Return list of potential anchor words from a sparse docwords matrix

    Candiate anchor words are words which appear in a significant number of
    documents. These are not rarewords persey (or else they would probably be
    filtered during pre-processing), but do not appear in enough documents to
    be useful as an anchor word.
    """
    candidate_anchors = []
    for i in range(M.shape[0]):
        if M[i, :].nnz > doc_threshold:
            candidate_anchors.append(i)
    return candidate_anchors


def find_anchors(Q, k, project_dim, candidates):
    """Uses stabalized Gram-Schmidt decomposition to find k anchors"""
    # don't modify the original Q
    Q = Q.copy()

    # normalized rows of Q and perform dimensionality reduction
    row_sums = Q.sum(1)
    for i in range(len(Q[:, 0])):
        Q[i, :] = Q[i, :] / float(row_sums[i])
    Q = random_projection(Q, project_dim)

    # setup book keeping for gram-schmidt
    anchors = numpy.zeros(k, dtype=numpy.int)
    basis = numpy.zeros((k-1, Q.shape[1]))

    # find the farthest point p1 from the origin
    max_dist = 0
    for i in candidates:
        dist = numpy.dot(Q[i], Q[i])
        if dist > max_dist:
            max_dist = dist
            anchors[0] = i

    # let p1 be the origin of our coordinate system
    for i in candidates:
        Q[i] = Q[i] - Q[anchors[0]]

    # find the farthest point from p1
    max_dist = 0
    for i in candidates:
        dist = numpy.dot(Q[i], Q[i])
        if dist > max_dist:
            max_dist = dist
            anchors[1] = i
            basis[0] = Q[i] / numpy.sqrt(numpy.dot(Q[i], Q[i]))

    # stabilized gram-schmidt to finds new anchor words to expand our subspace
    for j in range(1, k - 1):
        # project all the points onto our basis and find the farthest point
        max_dist = 0
        for i in candidates:
            Q[i] = Q[i] - numpy.dot(Q[i], basis[j-1]) * basis[j - 1]
            dist = numpy.dot(Q[i], Q[i])
            if dist > max_dist:
                max_dist = dist
                anchors[j + 1] = i
                basis[j] = Q[i] / numpy.sqrt(numpy.dot(Q[i], Q[i]))

    # return anchors as list
    return anchors.tolist()
