"""Implementations of other topic models for caparison"""

import gensim as gs

from . import topic


def gensim_lda(corpus, K, theta_attr=None, z_attr=None):
    bows = topic._gensim_bows(corpus)
    lda = gs.models.LdaModel(
        corpus=bows,
        num_topics=K,
    )
    if theta_attr or z_attr:
        topic._gensim_assign(corpus, bows, lda, theta_attr, z_attr)
    return lda.get_topics().T
