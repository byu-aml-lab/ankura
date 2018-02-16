"""Functions for using and displaying topics"""

import collections
import sys

import gensim as gs
import numpy as np
import scipy.spatial as sp


def topic_summary(topics, corpus=None, n=10):
    """Gets the top n tokens per topic.

    If a vocabulary is provided, the tokens are returned instead of the types.
    """
    summary = []
    for k in range(topics.shape[1]):
        index = []
        for word in np.argsort(topics[:, k])[-n:][::-1]:
            index.append(word)
        summary.append(index)

    if corpus:
        summary = [[corpus.vocabulary[w] for w in topic] for topic in summary]
    return summary


def lda_assign(corpus, topics, theta_attr=None, z_attr=None):
    """Assigns documents or tokens to topics using LDA with fixed topics

    If theta_attr is given, each document will be given a per-document topic
    distribution.  If z_attr is given, each document will be given a sequence
    of topic assignments corresponding to the tokens in the document. One or
    both of these metadata attribute names must be given.
    """
    if not theta_attr and not z_attr:
        raise ValueError('Either theta_attr or z_attr must be given')

    V, K = topics.shape
    lda = gs.models.LdaModel(
        num_topics=K,
        id2word={i: i for i in range(V)}, # LdaModel gets V from this dict
    )
    lda.state.sstats = topics.astype(lda.dtype).T * len(corpus.documents)
    lda.sync_state()

    bows = _gensim_bows(corpus)
    _gensim_assign(corpus, bows, lda, theta_attr, z_attr)


def _gensim_bows(corpus):
    bows = []
    for doc in corpus.documents:
        bow = collections.defaultdict(int)
        for t in doc.tokens:
            bow[t.token] += 1
        bows.append(bow)
    return [list(bow.items()) for bow in bows]


def _gensim_assign(corpus, bows, lda, theta_attr, z_attr):
    for doc, bow in zip(corpus.documents, bows):
        gamma, phi = lda.inference([bow], collect_sstats=z_attr)
        if theta_attr:
            doc.metadata[theta_attr] = gamma[0] / gamma[0].sum()
        if z_attr:
            w = [t.token for t in doc.tokens]
            doc.metadata[z_attr] = phi.argmax(axis=0)[w].tolist()



def cross_reference(corpus, theta_attr, xref_attr, n=sys.maxsize, threshold=1):
    """Finds the nearest documents by topic similarity.

    The documents of the corpus must include a metadata value for theta_attr
    giving a vector representation of the document. Typically, this is a topic
    distribution obtained with assign_topics. The vector representation is then
    used to compute distances between documents.

    For the purpose of choosing cross references, the closest n documents will
    be considered (default=sys.maxsize), although Documents whose similarity is
    behond the threshold (default=1) are excluded.  A threshold of 1 indicates
    that no filtering should be done, while a 0 indicates that only exact
    matches should be returned. The resulting cross references are stored on
    each document of the Corpus under the xref_attr.
    """
    for doc in corpus.documents:
        doc_theta = doc.metadata[theta_attr]
        dists = [sp.distance.cosine(doc_theta, d.metadata[theta_attr])
                 if doc is not d else float('nan')
                 for d in corpus.documents]
        dists = np.array(dists)
        xrefs = list(corpus.documents[i] for i in dists.argsort()[:n] if dists[i] <= threshold)
        doc.metadata[xref_attr] = xrefs


# TODO add classifier
