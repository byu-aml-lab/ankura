"""Functions for construction and modification of cooccurrence matrices"""

import scipy.sparse
import numpy


def build(corpus):
    """Constructs a cooccurrence matrix from a Corpus"""
    # See supplementary 4.1 of Aurora et al. 2012 for more details
    V, D = len(corpus.vocabulary), len(corpus.documents)
    data, row, col = [], [], []
    H_hat = numpy.zeros(V)

    for i, doc in enumerate(corpus.documents):
        n_d = len(doc.types)
        if n_d <= 1:
            continue

        norm = 1 / (n_d * (n_d - 1))
        sqrt_norm = numpy.sqrt(norm)

        for token in doc.types:
            data.append(sqrt_norm)
            row.append(token.type)
            col.append(i)
            H_hat[token.type] += norm

    H_tilde = scipy.sparse.coo_matrix((data, (row, col)), (V, D)).tocsc()
    Q = H_tilde * H_tilde.T - numpy.diag(H_hat)
    return numpy.array(Q / D)
