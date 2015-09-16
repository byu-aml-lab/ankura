"""Functions for constructing cooccurrence matrices used for topic recovery"""

import numpy

def construct_Q(M):
    """Constructs a cooccurrence matrix Q from a sparse docword matrix M

    The docword matrix M is expected to be a sparse scipy matrix which can be
    converted to csc form. The output matrix will be a 2d numpy array.
    """
    vocab_size, num_docs = M.shape

    # See supplementary 4.1 of Aurora et. al. 2012 for information on these
    H_tilde = M.tocsc()
    H_hat = numpy.zeros(vocab_size)

    # TODO Do this without indptr from csc, so any sparse matrix format works
    # Construct H_tilde and H_hat
    for j in xrange(H_tilde.indptr.size - 1):
        # get indices of column j
        col_start = H_tilde.indptr[j]
        col_end = H_tilde.indptr[j + 1]
        row_indices = H_tilde.indices[col_start: col_end]

        # get count of tokens in column (document) and compute norm
        count = numpy.sum(H_tilde.data[col_start: col_end])
        norm = count * (count - 1)

        # update H_hat and H_tilde (see supplementary)
        if norm != 0:
            H_hat[row_indices] = H_tilde.data[col_start: col_end] / norm
            H_tilde.data[col_start: col_end] /= numpy.sqrt(norm)


    # construct and return normalized Q
    Q = H_tilde * H_tilde.transpose() - numpy.diag(H_hat)
    return numpy.array(Q / num_docs)


# TODO add ways to augment Q matrix with additional labeled data
