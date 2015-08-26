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
