"""Functions for using and displaying topics"""

import collections
import sys

import numpy as np
import scipy.spatial

from . import util

try:
    import gensim
except ImportError:
    gensim = None


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
    if not gensim:
        _sampling_lda_assign(corpus, topics, theta_attr, z_attr)
        return

    # Convert corpus to gensim bag-of-words format
    bows = []
    for doc in corpus.documents:
        bow = collections.defaultdict(int)
        for t in doc.tokens:
            bow[t.token] += 1
        bows.append(bow)
    bows = [list(bow.items()) for bow in bows]

    # Build lda with fixed topics
    V, K = topics.shape
    lda = gensim.models.LdaModel(
        num_topics=K,
        id2word={i: i for i in range(V)}, # LdaModel gets V from this dict
    )
    lda.state.sstats = topics.astype(lda.dtype).T
    lda.sync_state()

    # Make topic assignments
    doc_topics = lda.get_document_topics(bows, per_word_topics=True)
    for doc, (sparse_theta, sparse_z, _) in zip(corpus.documents, doc_topics):
        if theta_attr:
            theta = np.zeros(K)
            for topic, prob in sparse_theta:
                theta[topic] = prob
            theta /= theta.sum()
            doc.metadata[theta_attr] = theta
        if z_attr:
            sparse_z= {word: topics[0] for word, topics in sparse_z}
            z = [sparse_z[t.token] for t in doc.tokens]
            doc.metadata[z_attr] = z


def _sampling_lda_assign(corpus, topics, theta_attr=None, z_attr=None, alpha=.01, num_iters=10):
    # Initialize counter and assignments
    _, K = topics.shape
    c = np.zeros((len(corpus.documents), K))
    z = [np.random.randint(K, size=len(d.tokens)) for d in corpus.documents]
    for d, z_d in enumerate(z):
        for z_dn in z_d:
            c[d, z_dn] += 1

    # Collapsed LDA Gibbs sampler with fixed topics
    for _ in range(num_iters):
        for d, (doc, z_d) in enumerate(zip(corpus.documents, z)):
            for n, w_dn in enumerate(doc.tokens):
                c[d, z_d[n]] -= 1
                cond = [alpha + c[d, t] * topics[w_dn.token, t] for t in range(K)]
                z_d[n] = util.sample_categorical(cond)
                c[d, z_d[n]] += 1

    # Make topic assignments
    if theta_attr:
        for doc, c_d in zip(corpus.documents, c):
            doc.metadata[theta_attr] = c_d / c_d.sum()
    if z_attr:
        for doc, z_d in zip(corpus.documents, z):
            doc.metadata[z_attr] = z_d.tolist()


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
        dists = [scipy.spatial.distance.cosine(doc_theta, d.metadata[theta_attr])
                 if doc is not d else float('nan')
                 for d in corpus.documents]
        dists = np.array(dists)
        xrefs = list(corpus.documents[i] for i in dists.argsort()[:n] if dists[i] <= threshold)
        doc.metadata[xref_attr] = xrefs


# TODO add classifier
