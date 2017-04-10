"""Functions for construction and modification of cooccurrence matrices"""

import numpy


def build(corpus):
    """Constructs a cooccurrence matrix from a Corpus"""
    # See supplementary 4.1 of Aurora et al. 2012 for more details
    V, D = len(corpus.vocabulary), len(corpus.documents)
    Q = numpy.zeros((V, D))
    for doc in corpus.documents:
        H_d = numpy.zeros(V)
        for token in doc.types:
            H_d[token.type] += 1
        n_d = len(doc.types)
        Q += (numpy.outer(H_d, H_d) - numpy.diag(H_d)) / (n_d * (n_d - 1))
    return Q / D
