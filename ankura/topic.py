"""Functions for using and displaying topics"""

import collections
import functools
import sys
import collections

import numpy as np
import scipy.spatial
import sklearn.decomposition
import gensim
import tqdm

from math import log, exp
from . import pipeline, util


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


def sampling_assign(corpus, topics, theta_attr=None, z_attr=None, alpha=.01, num_iters=10):
    """Predicts topic assignments for a corpus.

    Topic inference is done using Gibbs sampling with Latent Dirichlet
    Allocation and fixed topics following Griffiths and Steyvers 2004. The
    parameter alpha specifies a symetric Dirichlet prior over the document
    topic distributions. The parameter num_iters controlls how many iterations
    of sampling should be performed.

    If theta_attr is given, each document is given a metadata value describing
    the document-topic distribution as an array. If the z_attr is given, each
    document is given a metadata value describing the token level topic
    assignments. At east one of the attribute names must be given.

    """
    if not theta_attr and not z_attr:
        raise ValueError('Either theta_attr or z_attr must be given')

    T = topics.shape[1]

    c = np.zeros((len(corpus.documents), T))
    z = [np.random.randint(T, size=len(d.tokens)) for d in corpus.documents]
    for d, z_d in enumerate(z):
        for z_dn in z_d:
            c[d, z_dn] += 1

    for _ in range(num_iters):
        for d, (doc, z_d) in enumerate(zip(corpus.documents, z)):
            for n, w_dn in enumerate(doc.tokens):
                c[d, z_d[n]] -= 1
                cond = [alpha + c[d, t] * topics[w_dn.token, t] for t in range(T)]
                z_d[n] = util.sample_categorical(cond)
                c[d, z_d[n]] += 1

    if theta_attr:
        for doc, c_d in zip(corpus.documents, c):
            doc.metadata[theta_attr] = c_d / c_d.sum()
    if z_attr:
        for doc, z_d in zip(corpus.documents, z):
            doc.metadata[z_attr] = z_d.tolist()


def variational_assign(corpus, topics, theta_attr='theta', docwords_attr=None):
    """Predicts topic assignments for a corpus.

    Topic inference is done using online variational inference with Latent
    Dirichlet Allocation and fixed topics following Hoffman et al., 2010. Each
    document is given a metadata value named by theta_attr corresponding to the
    its predicted topic distribution.

    If docwords_attr is given, then the corpus metadata with that name is
    assumed to contain a pre-computed sparse docwords matrix. Otherwise, this
    docwords matrix will be recomputed.
    """
    V, K = topics.shape
    if docwords_attr:
        docwords = corpus.metadata[docwords_attr]
        if docwords.shape[1] != V:
            raise ValueError('Mismatch between topics and docwords shape')
    else:
        docwords = pipeline.build_docwords(corpus, V)

    lda = sklearn.decomposition.LatentDirichletAllocation(K)
    lda.components_ = topics.T
    lda._check_params()
    lda._init_latent_vars(V)
    theta = lda.transform(docwords)

    for doc, theta_d in zip(corpus.documents, theta):
        doc.metadata[theta_attr] = theta_d


def gensim_assign(corpus, topics, theta_attr=None, z_attr=None):
    if not theta_attr and not z_attr:
        raise ValueError('Either theta_attr or z_attr must be given')

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


def cross_reference(corpus, attr, doc=None, n=sys.maxsize, threshold=1):
    """Finds the nearest documents by topic similarity.

    The documents of the corpus must include a metadata value giving a vector
    representation of the document. Typically, this is a topic distribution
    obtained with an assign function and a metadata_attr. The vector
    representation is then used to compute distances between documents.

    If a document is given, then a list of references is returned for that
    document. Otherwise, cross references for each document in a corpus are
    given in a dict keyed by the documents. Consequently, the documents of the
    corpus must be hashable.

    The closest n documents will be returned (default=sys.maxsize). Documents
    whose similarity is behond the threshold (default=1) will not be returned.
    A threshold of 1 indicates that no filtering should be done, while a 0
    indicates that only exact matches should be returned.
    """
    def _xrefs(doc):
        doc_theta = doc.metadata[attr]
        dists = [scipy.spatial.distance.cosine(doc_theta, d.metadata[attr])
                 if doc is not d else float('nan')
                 for d in corpus.documents]
        dists = np.array(dists)
        return list(corpus.documents[i] for i in dists.argsort()[:n]
                    if dists[i] <= threshold)

    if doc:
        return _xrefs(doc)
    else:
        return [_xrefs(doc) for doc in corpus.documents]


def free_classifier(topics, Q, labels, epsilon=1e-7):
    """Creates a topic-based linear classifier. Details forthcoming..."""
    K = len(labels)
    V = Q.shape[0] - K

    # Smooth and column normalize class-topic weights
    A_f = topics[-K:] + epsilon
    A_f /= A_f.sum(axis=0)

    # class_given_word
    Q = Q / Q.sum(axis=1, keepdims=True) # row-normalize Q without original
    Q_L = Q[-K:, :V]

    @functools.wraps(free_classifier)
    def _classifier(doc, attr='theta'):
        H = np.zeros(V)
        for w_d in doc.tokens:
            H[w_d.token] += 1

        topic_score = A_f.dot(doc.metadata[attr])
        topic_score /= topic_score.sum(axis=0)

        word_score = Q_L.dot(H)
        word_score /= word_score.sum(axis=0)

        return labels[np.argmax(topic_score + word_score)]
    return _classifier


def free_classifier_revised(topics, Q, labels, epsilon=1e-7):
    """same as function above, with a few minor math fixes"""
    K = len(labels)
    V = Q.shape[0] - K

    # Smooth and column normalize class-topic weights
    A_f = topics[-K:] + epsilon
    A_f /= A_f.sum(axis=0)

    # class_given_word
    Q = Q / Q.sum(axis=1, keepdims=True) # row-normalize Q without original
    Q_L = Q[:V, -K:]

    @functools.wraps(free_classifier)
    def _classifier(doc, attr='theta'):
        H = np.zeros(V)
        for w_d in doc.tokens:
            H[w_d.token] += 1

        # normalize H
        H = H / H.sum(axis=0)

        topic_score = A_f.dot(doc.metadata[attr])
        topic_score /= topic_score.sum(axis=0)

        word_score = H.dot(Q_L)
        word_score /= word_score.sum(axis=0)

        return labels[np.argmax(topic_score + word_score)]
    return _classifier


def free_classifier_line_not_gibbs(corpus, attr_name, labeled_docs,
                            topics, C, labels, epsilon=1e-7):

    K = len(labels)

    # Smooth and column normalize class-topic weights
    A_f = topics[-K:] + epsilon
    A_f /= A_f.sum(axis=0)

    # column normalize topic-label matrix
    C_f = C[0:, -K:]
    C_f /= C_f.sum(axis=0)

    L = np.zeros(K)
    for d, doc in enumerate(corpus.documents):
        if d in labeled_docs:
            label_name = doc.metadata[attr_name]
            i = labels.index(label_name)
            L[i] += 1

    L = L / L.sum(axis=0) # normalize L to get the label probabilities

    @functools.wraps(free_classifier)
    def _classifier(doc, attr='z'):
        final_score = np.zeros(K)
        for i, l in enumerate(L):
            product = l
            doc_topic_count = collections.Counter(doc.metadata[attr])
            for topic, count in doc_topic_count.items():
                product *= C_f[topic, i]**count

            final_score[i] = product

        return labels[np.argmax(final_score)]
    return _classifier


def free_classifier_dream(corpus, attr_name, labeled_docs,
                            topics, C, labels, epsilon=1e-7):
    L = len(labels)

    # column-normalized word-topic matrix without labels
    A = topics[:-L]
    A /= A.sum(axis=0)

    _, K = A.shape # K is number of topics

    # column normalize topic-label matrix
    C_f = C[0:, -L:]
    C_f /= C_f.sum(axis=0)

    phi = np.zeros(L) # emperically observe labels
    for d, doc in enumerate(corpus.documents):
        if d in labeled_docs:
            label_name = doc.metadata[attr_name];
            i = labels.index(label_name)
            phi[i] += 1
    phi = phi / phi.sum(axis=0) # normalize phi to get the label probabilities

    @functools.wraps(free_classifier)
    def _classifier(doc):
        results = np.log(phi)
        for l in range(L):
            for n, w_n in enumerate(doc.tokens):
                results[l] += np.log(sum(C_f[t, l] * A[w_n.token, t] for t in range(K)))

        return labels[np.argmax(results)]
    return _classifier


def free_classifier_line_model(corpus, attr_name, labeled_docs,
                                    topics, C, labels, epsilon=1e-7, num_iters=10):

    L = len(labels)

    # column-normalized word-topic matrix without labels
    A = topics[:-L]
    A /= A.sum(axis=0)

    _, K = A.shape # K is number of topics

    # column normalize topic-label matrix
    C_f = C[0:, -L:]
    C_f /= C_f.sum(axis=0)

    phi = np.zeros(L) # emperically observe labels
    for d, doc in enumerate(corpus.documents):
        if d in labeled_docs:
            label_name = doc.metadata[attr_name];
            i = labels.index(label_name)
            phi[i] += 1
    phi = phi / phi.sum(axis=0) # normalize phi to get the label probabilities

    @functools.wraps(free_classifier)
    def _classifier(doc):
        l = np.random.randint(L)
        z = np.random.randint(K, size=len(doc.tokens))

        doc_topic_count = collections.Counter(z) # doc_topic_count maps topic assignments to counts
        for _ in range(num_iters):
            l_cond = np.log(phi) # not in log space: cond = phi
            for s in range(L):
                for topic, count in doc_topic_count.items():
                    l_cond[s] += count * np.log(C_f[topic, s]) # not in log space: cond[s] *= C_f[topic, s]**count
            l = util.sample_log_categorical(l_cond)

            for n, w_n in enumerate(doc.tokens):
                doc_topic_count[z[n]] -= 1
                z_cond = C_f[:K,l] * A[w_n.token,:K] # z_cond = [C_f[t, l] * A[w_n.token, t] for t in range(K)] # eq 2
                z[n] = util.sample_categorical(z_cond)
                doc_topic_count[z[n]] += 1

        return labels[l]
    return _classifier


def free_classifier_v_model(corpus, attr_name, labeled_docs,
                                    topics, labels, epsilon=1e-7, num_iters=10):

    L = len(labels)

    # column-normalized word-topic matrix without labels
    A = topics[:-L]
    A /= A.sum(axis=0)

    _, K = A.shape # K is number of topics

    # Smooth and column normalize class-topic weights
    A_f = topics[-L:] + epsilon
    A_f /= A_f.sum(axis=0)

    @functools.wraps(free_classifier)
    def _classifier(doc):
        l = np.random.randint(L)
        z = np.random.randint(K, size=len(doc.tokens))

        doc_topic_count = collections.Counter(z) # doc_topic_count maps topic assignments to counts
        for _ in range(num_iters):
            l_cond = [sum(A_f[x, topic]*count for topic, count in doc_topic_count.items()) for x in range(L)]

            l = util.sample_categorical(l_cond)
            B = l_cond[l] # B is a constant (summation of A_F[l, z_i])

            for n, w_n in enumerate(doc.tokens):
                B -= A_f[l, z[n]]
                z_cond = [A[w_n.token, t] * (A_f[l, t] + B) for t in range(K)]
                z[n] = util.sample_categorical(z_cond)
                B += A_f[l, z[n]]

        return labels[l]
    return _classifier
